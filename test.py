
        if not messages:
            return messages

        current_tokens = await count_messages_tokens_async(
            messages=messages, tools=tools, use_exact=False
        )
        if current_tokens <= max_tokens:
            return messages

        before_tokens = current_tokens
        before_count = len(messages)
        logger.warning(
            f"Agent context too large: {current_tokens} > {max_tokens}. "
            f"Starting safe trim."
        )

        # Шаг 1: укоротить длинные content
        shortened_messages: List[Dict[str, Any]] = []
        for msg in messages:
            msg_copy = dict(msg)
            msg_copy = self._truncate_message_content(
                msg_copy, settings.AGENT_TOOL_RESULT_TEXT_LIMIT
            )
            shortened_messages.append(msg_copy)

        current_tokens = await count_messages_tokens_async(
            messages=shortened_messages, tools=tools, use_exact=False
        )
        if current_tokens <= max_tokens:
            record_agent_trimming(
                before_tokens=before_tokens,
                after_tokens=current_tokens,
                messages_before=before_count,
                messages_after=len(shortened_messages),
                strategy="truncate_content",
            )
            return shortened_messages

        # Шаг 2: разбивка на блоки
        blocks = self._split_messages_into_blocks(shortened_messages)

        system_block: Optional[List[Dict[str, Any]]] = None
        first_user_block: Optional[List[Dict[str, Any]]] = None
        middle_blocks: List[List[Dict[str, Any]]] = []
        for block in blocks:
            first_msg = block[0]
            if system_block is None and first_msg.get("role") == "system":
                system_block = block
                continue
            if first_user_block is None and first_msg.get("role") == "user":
                first_user_block = block
                continue
            middle_blocks.append(block)

        protected_tail_count = min(3, len(middle_blocks))
        if protected_tail_count > 0:
            tail_blocks = middle_blocks[-protected_tail_count:]
            middle_blocks = middle_blocks[:-protected_tail_count]
        else:
            tail_blocks = []

        # Фиксированная часть (system + first user + tail) — токены считаем один раз
        fixed_messages: List[Dict[str, Any]] = []
        if system_block:
            fixed_messages.extend(system_block)
        if first_user_block:
            fixed_messages.extend(first_user_block)
        for b in tail_blocks:
            fixed_messages.extend(b)

        fixed_tokens = 0
        if fixed_messages:
            fixed_tokens = await count_messages_tokens_async(
                messages=fixed_messages, tools=tools, use_exact=False
            )

        # Токены каждого middle-блока — считаем один раз
        block_tokens: List[int] = []
        for b in middle_blocks:
            t = await count_messages_tokens_async(
                messages=b, tools=tools, use_exact=False
            )
            block_tokens.append(t)

        def build(indices: List[int]) -> List[Dict[str, Any]]:
            result: List[Dict[str, Any]] = list(fixed_messages)
            for i in indices:
                result.extend(middle_blocks[i])
            return result

        def total_for(indices: List[int]) -> int:
            return fixed_tokens + sum(block_tokens[i] for i in indices)

        protect_chunks = settings.AGENT_TRIM_PROTECT_CHUNK_BLOCKS

        # Шаг 3: удаление middle-блоков
        if protect_chunks:
            non_chunk_idx = [
                i for i, b in enumerate(middle_blocks)
                if not self._block_contains_chunks(b)
            ]
            chunk_idx = [
                i for i, b in enumerate(middle_blocks)
                if self._block_contains_chunks(b)
            ]

            # 3a. удаляем не-чанковые блоки (сначала самые старые)
            remaining_non_chunk = list(non_chunk_idx)
            while remaining_non_chunk:
                candidate_idx = remaining_non_chunk + chunk_idx
                if total_for(candidate_idx) <= max_tokens:
                    candidate = build(candidate_idx)
                    record_agent_trimming(
                        before_tokens=before_tokens,
                        after_tokens=total_for(candidate_idx),
                        messages_before=before_count,
                        messages_after=len(candidate),
                        strategy="drop_non_chunk_blocks",
                    )
                    return candidate
                remaining_non_chunk.pop(0)

            # 3b. все не-чанковые удалены, проверяем только чанковые
            candidate_idx = list(chunk_idx)
            if total_for(candidate_idx) <= max_tokens:
                candidate = build(candidate_idx)
                record_agent_trimming(
                    before_tokens=before_tokens,
                    after_tokens=total_for(candidate_idx),
                    messages_before=before_count,
                    messages_after=len(candidate),
                    strategy="drop_non_chunk_blocks",
                )
                return candidate

            # 3c. удаляем чанковые блоки (сначала самые старые)
            while candidate_idx:
                candidate_idx.pop(0)
                if total_for(candidate_idx) <= max_tokens:
                    candidate = build(candidate_idx)
                    record_agent_trimming(
                        before_tokens=before_tokens,
                        after_tokens=total_for(candidate_idx),
                        messages_before=before_count,
                        messages_after=len(candidate),
                        strategy="drop_chunk_blocks",
                    )
                    return candidate
        else:
            # Удаляем middle-блоки строго в порядке появления (старые → новые)
            remaining = list(range(len(middle_blocks)))
            while remaining:
                if total_for(remaining) <= max_tokens:
                    candidate = build(remaining)
                    record_agent_trimming(
                        before_tokens=before_tokens,
                        after_tokens=total_for(remaining),
                        messages_before=before_count,
                        messages_after=len(candidate),
                        strategy="drop_middle_blocks",
                    )
                    return candidate
                remaining.pop(0)

        # Шаг 4: агрессивная обрезка (все middle удалены, фиксированную часть режем жёстко)
        candidate = build([])
        aggressively_shortened = [
            self._truncate_message_content(dict(m), 1500)
            for m in candidate
        ]
        current_tokens = await count_messages_tokens_async(
            messages=aggressively_shortened, tools=tools, use_exact=False
        )
        if current_tokens <= max_tokens:
            record_agent_trimming(
                before_tokens=before_tokens,
                after_tokens=current_tokens,
                messages_before=before_count,
                messages_after=len(aggressively_shortened),
                strategy="aggressive_shorten",
            )
            return aggressively_shortened

        # Шаг 5: emergency — оставляем только system, first user и последнее сообщение
        emergency: List[Dict[str, Any]] = []
        if system_block:
            emergency.extend(system_block)
        if first_user_block:
            emergency.extend(first_user_block)
        if aggressively_shortened:
            emergency.append(aggressively_shortened[-1])
        logger.error(
            "Agent context emergency trim: оставлены только system, first user "
            "и последнее сообщение."
        )
        record_agent_trimming(
            before_tokens=before_tokens,
            after_tokens=0,
            messages_before=before_count,
            messages_after=len(emergency),
            strategy="emergency",
        )
        return emergency
