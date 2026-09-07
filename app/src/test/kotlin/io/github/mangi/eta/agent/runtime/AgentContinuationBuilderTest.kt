package io.github.mangi.eta.agent.runtime

import io.github.mangi.eta.agent.model.AgentModelClient
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class AgentContinuationBuilderTest {
    @Test
    fun continuationPreservesPromptImagesAndFullTranscript() {
        val image = AgentModelClient.ModelImage(
            reference = "data:image/png;base64,AA==",
            mimeType = "image/png",
            bytes = 1,
        )
        val request = AgentRuntimeWire.RunRequest(
            runId = "run-old",
            prompt = "观察屏幕",
            config = modelConfig(),
            images = listOf(image),
            history = listOf(AgentModelClient.ConversationMessage(role = "user", content = "更早的问题")),
            handoff = AgentRuntimeWire.EntryHandoff(
                id = "run-old",
                source = "agent_ui",
                payload = AgentUiHandoffPayload(conversationId = "conversation-1").toJson(),
            ),
        )
        val response = AgentModelClient.ModelResponse.Text(
            content = "完成",
            transcript = listOf(
                AgentModelClient.ConversationMessage(
                    role = "assistant",
                    toolCallsJson = "[{\"id\":\"call-1\",\"type\":\"function\",\"function\":{\"name\":\"observe_screen\",\"arguments\":\"{}\"}}]",
                ),
                AgentModelClient.ConversationMessage(
                    role = "tool",
                    toolCallId = "call-1",
                    content = "{\"ok\":true}",
                ),
                AgentModelClient.ConversationMessage(role = "assistant", content = "完成"),
            ),
        )

        val continuation = AgentContinuationBuilder.build(
            request = request,
            response = response,
            supplement = "继续检查",
            newRunId = "run-next",
            createdAt = 123L,
        )

        assertEquals("run-next", continuation.runId)
        assertEquals("继续检查", continuation.prompt)
        assertTrue(continuation.images.isEmpty())
        assertEquals(
            listOf("user", "user", "assistant", "tool", "assistant"),
            continuation.history.map { it.role },
        )
        assertTrue(continuation.history[1].contentJson.contains("未写入持久会话"))
        assertTrue(!continuation.history[1].contentJson.contains("base64"))
        assertEquals("call-1", continuation.history[3].toolCallId)
        assertEquals("run-next", continuation.handoff?.id)
        val payload = AgentUiHandoffPayload.from(continuation.handoff?.payload.orEmpty())
        assertEquals("conversation-1", payload.conversationId)
        assertEquals(
            AgentUiHandoffPayload.Supplement(
                index = 1,
                text = "继续检查",
                createdAt = 123L,
            ),
            payload.promptSupplement,
        )
        assertTrue(payload.supplements.isEmpty())
    }

    @Test
    fun continuationUsesHistoryReplacementBeforeTranscriptAndNewSupplementPrompt() {
        val request = AgentRuntimeWire.RunRequest(
            runId = "run-old",
            prompt = "原始问题",
            config = modelConfig(),
            images = emptyList(),
            history = listOf(
                AgentModelClient.ConversationMessage(role = "user", content = "未压缩旧历史"),
            ),
            handoff = AgentRuntimeWire.EntryHandoff(
                id = "run-old",
                source = "agent_ui",
                payload = AgentUiHandoffPayload(conversationId = "conversation-1").toJson(),
            ),
        )
        val replacement = listOf(
            AgentModelClient.ConversationMessage(role = "system", content = "压缩后的摘要"),
            AgentModelClient.ConversationMessage(role = "user", content = "保留的最新问题"),
        )
        val transcript = listOf(
            AgentModelClient.ConversationMessage(role = "assistant", content = "压缩前的最终回答"),
        )
        val response = AgentModelClient.ModelResponse.Text(
            content = "压缩前的最终回答",
            transcript = transcript,
            historyReplacement = replacement,
        )

        val continuation = AgentContinuationBuilder.build(
            request = request,
            response = response,
            supplement = "继续验证压缩后的上下文",
            newRunId = "run-next",
            createdAt = 456L,
        )

        assertEquals(replacement + transcript, continuation.history)
        assertTrue(continuation.history.none { it.content == "未压缩旧历史" })
        assertEquals("继续验证压缩后的上下文", continuation.prompt)
        val payload = AgentUiHandoffPayload.from(continuation.handoff?.payload.orEmpty())
        assertEquals(
            AgentUiHandoffPayload.Supplement(
                index = 1,
                text = "继续验证压缩后的上下文",
                createdAt = 456L,
            ),
            payload.promptSupplement,
        )
    }

    private fun modelConfig(): AgentModelClient.ModelConfig =
        AgentModelClient.ModelConfig(
            baseUrl = "https://example.invalid/v1",
            apiKey = "test-key",
            model = "test-model",
            systemPrompt = "",
        )
}
