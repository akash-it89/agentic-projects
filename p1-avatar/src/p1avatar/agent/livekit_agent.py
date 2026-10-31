"""Real-time voice agent on LiveKit (phases 5 and 6).

    livekit-server --dev                                   # in another terminal
    uv sync --group rag --group realtime                   # add --group nova for Nova Sonic
    uv run python -m p1avatar.agent.livekit_agent dev

AGENT_MODE=cascaded  Speaches (STT) -> Ollama (LLM) -> Kokoro (TTS), all local
AGENT_MODE=nova      Amazon Nova Sonic speech-to-speech; the same RAG tool is plugged in

Written against LiveKit Agents 1.x. Plugin argument names change between releases, so check them
against https://docs.livekit.io/agents/ for the version uv installed.

Not included yet: driving the Ditto avatar in real time. LiveKit's "virtual avatar" pattern runs a
separate worker that receives the agent's audio and publishes a video track; see the avatar examples
in github.com/livekit/agents. The turn-based kiosk (api/main.py) already shows the avatar.
"""

import asyncio
import json
import logging

from livekit.agents import Agent, AgentSession, JobContext, RunContext, WorkerOptions, cli, function_tool

from ..config import get_settings
from ..lang import LLM_LANGUAGE, default_voice, detect_lang
from ..rag.search import search_airport_info as rag_search

log = logging.getLogger("airport-agent")

INSTRUCTIONS = (
    "You are the voice assistant on an airport information kiosk. "
    "ALWAYS call search_airport_info before answering any question about the airport. "
    "Answer only with facts from its result. If it returns found=false, or the passages do not answer the "
    "question, say you don't have that information and suggest the airport information desk. Never guess. "
    "Reply in the same language the visitor spoke ("
    + ", ".join(LLM_LANGUAGE.values())
    + "), in at most two short sentences, with no lists."
)


class AirportAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=INSTRUCTIONS)

    @function_tool
    async def search_airport_info(self, context: RunContext, query: str) -> str:
        """Look up airport facts: gates, terminals, lounges, baggage, transport, Wi-Fi and services.

        Args:
            query: the visitor's question, in their own words and language.
        """
        # Retrieval is synchronous and model-heavy; keep it off the event loop so audio keeps flowing.
        result = await asyncio.to_thread(rag_search, query)
        log.info("search %r -> found=%s", query, result["found"])
        return json.dumps(result, ensure_ascii=False)


def build_session() -> AgentSession:
    s = get_settings()
    if s.agent_mode == "nova":
        from livekit.plugins import aws

        # Region and credentials come from the AWS environment; voice and model options per plugin docs.
        return AgentSession(llm=aws.realtime.RealtimeModel())

    from livekit.plugins import openai, silero

    return AgentSession(
        vad=silero.VAD.load(),
        # detect_language: otherwise the plugin assumes English. Verify the argument name for your version.
        stt=openai.STT(base_url=s.speaches_base_url, api_key="local", model=s.speaches_model, detect_language=True),
        llm=openai.LLM(base_url=s.ollama_base_url, api_key="ollama", model=s.ollama_model, temperature=0),
        tts=openai.TTS(base_url=s.kokoro_base_url, api_key="local", model="kokoro", voice=default_voice("en")),
    )


async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect()
    session = build_session()

    if get_settings().agent_mode == "cascaded":
        @session.on("user_input_transcribed")
        def _switch_voice(ev) -> None:
            # Kokoro voices are per language, so match the voice to what the visitor just said.
            if not getattr(ev, "is_final", True):
                return
            lang = detect_lang(getattr(ev, "transcript", "") or "")
            if lang and hasattr(session.tts, "update_options"):
                session.tts.update_options(voice=default_voice(lang))

    await session.start(room=ctx.room, agent=AirportAgent())
    await session.generate_reply(instructions="Greet the visitor in one short sentence in English, Hindi and Spanish.")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
