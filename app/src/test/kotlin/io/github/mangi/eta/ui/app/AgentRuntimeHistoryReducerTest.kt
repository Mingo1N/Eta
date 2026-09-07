package io.github.mangi.eta.ui.app

import io.github.mangi.eta.agent.model.AgentModelClient
import io.github.mangi.eta.ui.model.AgentChatUiState
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class AgentRuntimeHistoryReducerTest {
    @Test
    fun recoveryThenLiveDeliveryCommitsTranscriptOnlyOnce() {
        val initial = AgentChatUiState(
            messages = emptyList(),
            input = "",
            isStreaming = true,
            thinkingEnabled = false,
        )
        val transcript = listOf(
            AgentModelClient.ConversationMessage(role = "assistant", content = "完成")
        )

        val recovered = AgentRuntimeHistoryReducer.apply(initial, "run-1", transcript)
        val live = AgentRuntimeHistoryReducer.apply(recovered.state, "run-1", transcript)

        assertTrue(live.alreadyApplied)
        assertEquals(transcript, live.state.history)
        assertEquals(listOf("run-1"), live.state.appliedRuntimeRunIds)
    }

    @Test
    fun historyReplacementReplacesOldHistoryAndReplayIsIdempotent() {
        val oldHistory = listOf(
            AgentModelClient.ConversationMessage(role = "user", content = "旧历史"),
        )
        val replacement = listOf(
            AgentModelClient.ConversationMessage(role = "system", content = "压缩摘要"),
            AgentModelClient.ConversationMessage(role = "user", content = "保留问题"),
        )
        val transcript = listOf(
            AgentModelClient.ConversationMessage(role = "assistant", content = "压缩后回答"),
        )
        val initial = AgentChatUiState(
            messages = emptyList(),
            history = oldHistory,
            input = "",
            isStreaming = true,
            thinkingEnabled = false,
        )

        val recovered = AgentRuntimeHistoryReducer.apply(
            state = initial,
            runId = "run-compacted",
            additions = transcript,
            historyReplacement = replacement,
        )

        assertEquals(replacement + transcript, recovered.state.history)
        assertTrue(recovered.state.history.none { it.content == "旧历史" })
        assertEquals(listOf("run-compacted"), recovered.state.appliedRuntimeRunIds)

        val replay = AgentRuntimeHistoryReducer.apply(
            state = recovered.state,
            runId = "run-compacted",
            additions = transcript,
            historyReplacement = replacement,
        )

        assertTrue(replay.alreadyApplied)
        assertEquals(recovered.state, replay.state)
    }
}
