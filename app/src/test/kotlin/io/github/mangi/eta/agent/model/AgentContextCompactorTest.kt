package io.github.mangi.eta.agent.model

import io.github.mangi.eta.agent.runtime.AgentRunCancelledException
import io.github.mangi.eta.agent.runtime.AgentRunController
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertSame
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

class AgentContextCompactorTest {
    @Test
    fun belowBudgetHistoryIsReturnedUnchanged() {
        val history = listOf(
            message("user", "短问题"),
            message("assistant", "短回答"),
        )
        val provider = RecordingProvider()

        val result = compact(
            config = config(contextWindow = 10_000),
            history = history,
            provider = provider,
        )

        assertFalse(result.compacted)
        assertFalse(result.usedModelSummary)
        assertSame(history, result.history)
        assertTrue(provider.requests.isEmpty())
    }

    @Test
    fun configuredWindowUsesPercentagesInsteadOfMultiplyingByRawRatio() {
        val provider = RecordingProvider()
        val history = listOf(message("user", "x".repeat(3_000)))

        val result = compact(
            config = config(contextWindow = 1_000),
            history = history,
            provider = provider,
        )

        assertTrue(result.compacted)
        assertEquals(1, provider.requests.size)
    }

    @Test
    fun unknownWindowDefaultsTo128kCharactersPerTokenBudget() {
        val result = compact(
            config = config(contextWindow = null),
            history = emptyList(),
            currentUserMessage = JSONObject()
                .put("role", "user")
                .put("content", "x".repeat(360_000)),
            provider = RecordingProvider(),
        )

        assertTrue(result.compacted)
    }

    @Test
    fun compactionKeepsNewestCompleteRoundsInOrderAndStartsWithSystemSummary() {
        val provider = RecordingProvider(response = response("模型摘要"))
        // 预算算术：三个回合序列化后 ≈ 20.4k 字符；窗口 6800 → 字符预算 27.2k，
        // 触发线 19.04k（会压缩），目标 13.6k → historyBudget ≈ 13.5k 正好容纳
        // middle+new（≈12.2k）而装不下 old，因此期望保留的是后两个回合。
        val history = listOf(
            message("user", "old-user-" + "o".repeat(4_000)),
            message("assistant", "old-assistant-" + "o".repeat(4_000)),
            message("user", "middle-user-" + "m".repeat(4_000)),
            message("assistant", "middle-assistant-" + "m".repeat(4_000)),
            message("user", "new-user-" + "n".repeat(2_000)),
            message("assistant", "new-assistant-" + "n".repeat(2_000)),
            message("tool", "new-tool-result", toolCallId = "call-new"),
        )
        val systemMessages = JSONArray().put(
            JSONObject().put("role", "system").put("content", "system instruction"),
        )

        val result = compact(
            config = config(contextWindow = 6_800),
            history = history,
            systemMessages = systemMessages,
            provider = provider,
        )

        assertTrue(result.compacted)
        assertTrue(result.usedModelSummary)
        assertEquals("system", result.history.first().role)
        assertTrue(result.history.first().content.contains("模型摘要"))
        assertFalse(result.history.drop(1).firstOrNull()?.role == "tool")
        val tail = result.history.drop(1)
        assertEquals(
            listOf("user", "assistant", "user", "assistant", "tool"),
            tail.map { it.role },
        )
        assertTrue(tail.any { it.content.startsWith("middle-user-") })
        assertTrue(tail.any { it.content.startsWith("new-user-") })
        assertFalse(tail.any { it.content.startsWith("old-user-") })
        assertEquals(1, systemMessages.length())
    }

    @Test
    fun oversizedNewestRoundRemovesToolMessagesAndClearsStructuredCallsBeforeTruncating() {
        val provider = RecordingProvider(response = response("摘要"))
        val hugeStructuredContent = JSONObject()
            .put("type", "text")
            .put("text", "structured")
            .toString()
        val history = listOf(
            message("user", "old"),
            message("assistant", "old answer"),
            message("user", "new-user-" + "u".repeat(2_000)),
            message(
                role = "assistant",
                content = "new-assistant-" + "a".repeat(2_000),
                contentJson = hugeStructuredContent,
                toolCallsJson = toolCalls("call-new"),
            ),
            message("tool", "tool result that must not remain", toolCallId = "call-new"),
        )

        val result = compact(
            config = config(contextWindow = 500),
            history = history,
            provider = provider,
        )

        assertTrue(result.compacted)
        val retained = result.history.drop(1)
        assertEquals(listOf("user", "assistant"), retained.map { it.role })
        val assistant = retained.last()
        assertTrue(assistant.content.length < 2_000)
        assertEquals("", assistant.contentJson)
        assertEquals("", assistant.toolCallsJson)
        assertEquals("", assistant.toolCallId)
    }

    @Test
    fun successfulSummaryUsesSameProviderEmptyToolsAndNoopEvents() {
        val provider = RecordingProvider(response = response("安全摘要"))
        val config = config(contextWindow = 1_000)

        val result = compact(
            config = config,
            history = listOf(message("user", "x".repeat(3_000))),
            provider = provider,
        )

        assertTrue(result.compacted)
        assertTrue(result.usedModelSummary)
        assertEquals(config, provider.requests.single().config)
        assertEquals(0, provider.requests.single().tools.length())
        assertEquals(listOf("system", "user"), provider.requests.single().messages.roles())
        provider.callbacks.single().invoke(ProviderEvent.RequestStarted)
        assertTrue(result.history.first().content.contains("安全摘要"))
    }

    @Test
    fun ordinarySummaryFailureFallsBackToLocalSummary() {
        val provider = RecordingProvider(failure = IllegalStateException("provider failed"))

        val result = compact(
            config = config(contextWindow = 1_000),
            history = listOf(message("user", "x".repeat(3_000))),
            provider = provider,
        )

        assertTrue(result.compacted)
        assertFalse(result.usedModelSummary)
        assertTrue(result.history.first().content.contains("此前部分历史已压缩"))
    }

    @Test
    fun summaryAcceptsOnlyNonEmptyPlainEndTurnText() {
        listOf(
            response(content = "", finishReason = "stop"),
            response(content = "摘要", finishReason = "tool_calls"),
            response(
                content = JSONArray().put(JSONObject().put("type", "text").put("text", "摘要")),
                finishReason = "stop",
            ),
            response(
                content = "摘要",
                finishReason = "stop",
                toolCalls = JSONArray(),
            ),
        ).forEach { invalidResponse ->
            val provider = RecordingProvider(response = invalidResponse)
            val result = compact(
                config = config(contextWindow = 1_000),
                history = listOf(message("user", "x".repeat(3_000))),
                provider = provider,
            )
            assertFalse(result.usedModelSummary)
            assertTrue(result.history.first().content.contains("此前部分历史已压缩"))
        }
    }

    @Test
    fun summaryInputOmitsImagesAndResponsesOpaqueFields() {
        val image = "data:image/png;base64,SECRET_BASE64_PAYLOAD"
        // 历史必须超过 1000 token 窗口的触发线（4000 字符 × 70%）才会进入压缩，
        // 因此这里给 assistant 正文补足长度，否则 compactIfNeeded 直接原样返回。
        val filler = "p".repeat(3_000)
        val content = JSONObject()
            .put("type", "message")
            .put("text", "safe text")
            .put("image_url", JSONObject().put("url", image))
            .put("_eta_responses_output_items", JSONArray().put(
                JSONObject().put("encrypted_content", "OPAQUE_SECRET"),
            ))
            .put("previous_response_id", "RESPONSES_SECRET")
            .toString()
        val provider = RecordingProvider(response = response("摘要"))

        compact(
            config = config(contextWindow = 1_000),
            history = listOf(
                message(
                    role = "user",
                    content = content,
                    contentJson = JSONObject()
                        .put("image_url", image)
                        .put("encrypted_content", "OPAQUE_CONTENT_JSON")
                        .toString(),
                ),
                message(
                    role = "assistant",
                    content = "普通回答-" + filler,
                    toolCallsJson = toolCalls("call-safe", arguments = "OPAQUE_ARGUMENT"),
                ),
            ),
            provider = provider,
        )

        val summaryRequest = provider.requests.single().messages.toString()
        listOf(
            "data:image",
            "SECRET_BASE64_PAYLOAD",
            "OPAQUE_SECRET",
            "RESPONSES_SECRET",
            "OPAQUE_CONTENT_JSON",
            "OPAQUE_ARGUMENT",
            "encrypted_content",
            "_eta_responses_output_items",
            "previous_response_id",
        ).forEach { forbidden ->
            assertFalse("泄漏了 $forbidden", summaryRequest.contains(forbidden))
        }
        assertTrue(summaryRequest.contains("safe text"))
    }

    @Test
    fun existingSummariesAreMergedWithinBound() {
        val oldBody = "OLD_SUMMARY_" + "o".repeat(7_500)
        val history = listOf(
            message("system", "<eta_context_summary>\n$oldBody\n</eta_context_summary>"),
            message("user", "new history " + "h".repeat(3_000)),
        )
        val provider = RecordingProvider(response = response("NEW_SUMMARY_" + "n".repeat(7_500)))

        val result = compact(
            config = config(contextWindow = 1_000),
            history = history,
            provider = provider,
        )

        val summary = result.history.first()
        assertTrue(summary.content.length <= 8_050)
        assertTrue(summary.content.contains("OLD_SUMMARY_"))
        assertTrue(summary.content.contains("NEW_SUMMARY_"))
    }

    @Test
    fun cancellationFromSummaryProviderIsPropagated() {
        val provider = RecordingProvider(failure = AgentRunCancelledException())

        assertThrows(AgentRunCancelledException::class.java) {
            compact(
                config = config(contextWindow = 1_000),
                history = listOf(message("user", "x".repeat(3_000))),
                provider = provider,
            )
        }
    }

    @Test
    fun staleToolResultsStayIntactBelowBudget() {
        val messages = turnMessages(1, 20_000, contextMarker = "keep")

        val degraded = AgentContextCompactor.degradeStaleToolResults(
            config = config(contextWindow = 100_000),
            messages = messages,
            tools = JSONArray(),
        )

        assertEquals(0, degraded)
        assertEquals(20_000 + "keep-0-".length, messages.getJSONObject(1).getString("content").length)
    }

    @Test
    fun onlyStaleToolResultsAreTruncatedAndPairingSurvives() {
        val messages = turnMessages(4, 20_000)

        val degraded = AgentContextCompactor.degradeStaleToolResults(
            config = config(contextWindow = 2_000),
            messages = messages,
            tools = JSONArray(),
        )

        // 4 个回合里最近两个受保护，因此最多只有最早两个会被截断。
        assertEquals(2, degraded)
        listOf(1 to "turn-0-", 3 to "turn-1-").forEach { (index, marker) ->
            val content = messages.getJSONObject(index).getString("content")
            assertTrue("$marker 未被截断", content.length < 5_000)
            assertTrue(content.contains("重新调用该工具"))
            assertTrue(content.startsWith(marker))
        }
        listOf(5 to "turn-2-", 7 to "turn-3-").forEach { (index, marker) ->
            assertTrue("$marker 不应被截断", messages.getJSONObject(index).getString("content").startsWith(marker))
            assertEquals(20_000 + marker.length, messages.getJSONObject(index).getString("content").length)
        }
        // 消息一条没少，assistant 的 tool_calls 与 tool_call_id 配对保持完整。
        assertEquals(
            listOf("assistant", "tool", "assistant", "tool", "assistant", "tool", "assistant", "tool"),
            (0 until messages.length()).map { messages.getJSONObject(it).getString("role") },
        )
        assertEquals("call-0", messages.getJSONObject(1).getString("tool_call_id"))
        assertTrue(messages.getJSONObject(0).getJSONArray("tool_calls").getJSONObject(0).has("function"))
    }

    @Test
    fun nonTextToolContentIsNotRewritten() {
        val messages = JSONArray()
            .put(
                JSONObject()
                    .put("role", "assistant")
                    .put("content", "")
                    .put("tool_calls", JSONArray().put(toolCallJson("call-1"))),
            )
            .put(
                JSONObject()
                    .put("role", "tool")
                    .put("tool_call_id", "call-1")
                    .put("content", JSONArray().put(JSONObject().put("type", "image_url"))),
            )
            .put(
                JSONObject()
                    .put("role", "assistant")
                    .put("content", "")
                    .put("tool_calls", JSONArray().put(toolCallJson("call-2"))),
            )
            .put(
                JSONObject()
                    .put("role", "tool")
                    .put("tool_call_id", "call-2")
                    .put("content", "y".repeat(20_000)),
            )
            .put(
                JSONObject()
                    .put("role", "assistant")
                    .put("content", "")
                    .put("tool_calls", JSONArray().put(toolCallJson("call-3"))),
            )
            .put(
                JSONObject()
                    .put("role", "tool")
                    .put("tool_call_id", "call-3")
                    .put("content", "z".repeat(20_000)),
            )

        val degraded = AgentContextCompactor.degradeStaleToolResults(
            config = config(contextWindow = 2_000),
            messages = messages,
            tools = JSONArray(),
        )

        // 三个回合中只有最早一个可降级；其中的多模态 tool 正文不是字符串，必须原样保留。
        assertEquals(1, degraded)
        assertTrue(messages.getJSONObject(1).opt("content") is JSONArray)
        assertTrue(messages.getJSONObject(3).getString("content").length < 5_000)
        assertEquals(20_000, messages.getJSONObject(5).getString("content").length)
    }

    private fun turnMessages(turns: Int, resultChars: Int, contextMarker: String = "turn"): JSONArray =
        JSONArray().apply {
            repeat(turns) { index ->
                val callId = "call-$index"
                put(
                    JSONObject()
                        .put("role", "assistant")
                        .put("content", "")
                        .put("tool_calls", JSONArray().put(toolCallJson(callId))),
                )
                put(
                    JSONObject()
                        .put("role", "tool")
                        .put("tool_call_id", callId)
                        .put("content", "$contextMarker-$index-" + "x".repeat(resultChars)),
                )
            }
        }

    private fun toolCallJson(id: String): JSONObject =
        JSONObject()
            .put("id", id)
            .put("type", "function")
            .put("function", JSONObject().put("name", "test_tool").put("arguments", "{}"))

    private fun compact(
        config: AgentModelClient.ModelConfig,
        history: List<AgentModelClient.ConversationMessage>,
        provider: RecordingProvider,
        systemMessages: JSONArray = JSONArray(),
        currentUserMessage: JSONObject = JSONObject().put("role", "user").put("content", "current"),
        tools: JSONArray = JSONArray(),
    ): AgentContextCompactor.Result = AgentContextCompactor.compactIfNeeded(
        config = config,
        history = history,
        systemMessages = systemMessages,
        currentUserMessage = currentUserMessage,
        tools = tools,
        provider = provider,
        controller = AgentRunController(),
    )

    private fun config(contextWindow: Int?): AgentModelClient.ModelConfig =
        AgentModelClient.ModelConfig(
            baseUrl = "https://example.invalid/v1",
            apiKey = "test-key",
            model = "test-model",
            systemPrompt = "system",
            contextWindow = contextWindow,
        )

    private fun message(
        role: String,
        content: String,
        contentJson: String = "",
        toolCallId: String = "",
        reasoningContent: String = "",
        toolCallsJson: String = "",
    ) = AgentModelClient.ConversationMessage(
        role = role,
        content = content,
        contentJson = contentJson,
        toolCallId = toolCallId,
        reasoningContent = reasoningContent,
        toolCallsJson = toolCallsJson,
    )

    private fun toolCalls(id: String, arguments: String = "{}"): String = JSONArray().put(
        JSONObject()
            .put("id", id)
            .put("type", "function")
            .put(
                "function",
                JSONObject().put("name", "test_tool").put("arguments", arguments),
            ),
    ).toString()

    private fun response(
        content: Any? = "摘要",
        finishReason: String = "stop",
        toolCalls: JSONArray? = null,
    ): ProviderResponse {
        val assistant = JSONObject()
            .put("role", "assistant")
            .put("content", content)
            .put("finish_reason", finishReason)
        if (toolCalls != null) assistant.put("tool_calls", toolCalls)
        return ProviderResponse(assistant)
    }

    private class RecordingProvider(
        private val response: ProviderResponse? = null,
        private val failure: Exception? = null,
    ) : AgentProviderClient {
        override val id: String = "recording-provider"
        override val capabilities = ProviderCapabilities(
            endpoint = EndpointKind.CHAT_COMPLETIONS,
            streamingText = true,
            streamingToolCalls = true,
            imageInput = true,
            toolResultImages = false,
            strictTools = false,
            parallelToolCalls = false,
        )
        val requests = mutableListOf<ProviderRequest>()
        val callbacks = mutableListOf<(ProviderEvent) -> Unit>()

        override fun complete(
            request: ProviderRequest,
            runController: AgentRunController,
            onEvent: (ProviderEvent) -> Unit,
        ): ProviderResponse {
            requests += request
            callbacks += onEvent
            failure?.let { throw it }
            return response ?: ProviderResponse(
                JSONObject()
                    .put("role", "assistant")
                    .put("content", "摘要")
                    .put("finish_reason", "stop"),
            )
        }
    }

    private fun JSONArray.roles(): List<String> =
        (0 until length()).map { index -> getJSONObject(index).getString("role") }
}
