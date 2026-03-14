import os
import json
import asyncio
import logging
import pyaudio
import websockets
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEEPGRAM_KEY = os.getenv("DEEPGRAM_STT_KEY")
TTS_KEY = os.getenv("DEEPGRAM_TTS_KEY")

FLUX_URL = (
    "wss://api.deepgram.com/v1/listen"
    "?model=nova-2"
    "&encoding=linear16"
    "&sample_rate=16000"
    "&channels=1"
    "&endpointing=300"
)

class VoiceAgent:

    def __init__(self):

        self.llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model="llama-3.1-8b-instant",
            temperature=0.3
        )

        # _last_intent is set by run_flux() once the patient responds.
        # The LangGraph thread bridge reads this attribute to get YES/NO.
        self._last_intent = None

        logger.info("Voice Agent initialized")

    # ---------------------------------------------------
    # LLM reasoning
    # ---------------------------------------------------

    def think(self, user_text):

        print("\n🧠 Sending to LLM...")

        prompt = f"""
You are a healthcare claims assistant.

User said:
{user_text}

Classify the answer as YES or NO.

YES means the user agrees to correct the claim.
NO means the user refuses.

Respond with only YES or NO.
"""

        response = self.llm.invoke(prompt)

        result = response.content.strip().upper()

        print("🧠 LLM result:", result)

        return result

    # ---------------------------------------------------
    # TTS
    # ---------------------------------------------------

    async def speak(self, text):

        print("\n🔊 TTS START")

        uri = "wss://api.deepgram.com/v1/speak?model=aura-asteria-en&encoding=linear16&sample_rate=16000"

        headers = {
            "Authorization": f"Token {TTS_KEY}"
        }

        async with websockets.connect(uri, additional_headers=headers) as ws:

            print("🔊 Connected to TTS")

            await ws.send(json.dumps({
                "type": "Speak",
                "text": text
            }))

            await ws.send(json.dumps({"type": "Flush"}))

            p = pyaudio.PyAudio()

            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                output=True
            )

            while True:

                message = await ws.recv()

                if isinstance(message, bytes):

                    stream.write(message)

                else:

                    data = json.loads(message)

                    if data.get("type") == "Flushed":
                        break

            stream.stop_stream()
            stream.close()
            p.terminate()

            print("🔊 TTS FINISHED")

    # ---------------------------------------------------
    # Send correction back to Agent4
    # ---------------------------------------------------

    def send_to_agent4(self, claim):

        print("\n📤 Sending correction request to Agent4...")

        output = {
            "claim_id": claim["claim_id"],
            "status": "CORRECTION_REQUESTED",
            "original_decision": claim["decision"],
            "reason": "User approved correction via voice agent"
        }

        file = "../agent4/corrections_from_agent5.json"

        if os.path.exists(file):

            with open(file) as f:
                data = json.load(f)

        else:
            data = []

        data.append(output)

        with open(file, "w") as f:
            json.dump(data, f, indent=2)

        print("  Sent to Agent4")

    # ---------------------------------------------------
    # Voice conversation
    # ---------------------------------------------------

    async def run_flux(self, claim):

        print("\n🌐 Connecting to STT...")

        headers = {
            "Authorization": f"Token {DEEPGRAM_KEY}"
        }

        async with websockets.connect(FLUX_URL, additional_headers=headers) as ws:

            print("🌐 STT Connected")

            p = pyaudio.PyAudio()

            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=1024
            )

            stop_audio = asyncio.Event()
            transcript_received = asyncio.Event()

            print("🎤 Speak now...")

            # -------------------------
            # send microphone audio
            # -------------------------

            async def send_audio():

                while not stop_audio.is_set():

                    try:

                        data = stream.read(1024, exception_on_overflow=False)

                        await ws.send(data)

                    except Exception:
                        break

            # -------------------------
            # receive transcripts
            # -------------------------

            async def receive_transcript():

                async for message in ws:

                    data = json.loads(message)

                    if data.get("type") == "Results":

                        transcript = data["channel"]["alternatives"][0]["transcript"]
                        speech_final = data.get("speech_final", False)

                        if transcript and speech_final:

                            print("\n🗣️ USER:", transcript)

                            transcript_received.set()

                            decision = self.think(transcript)

                            # Store intent so LangGraph bridge can read it
                            self._last_intent = decision

                            if "YES" in decision:

                                await self.speak("Thank you. We will correct your claim.")

                                self.send_to_agent4(claim)

                            else:

                                await self.speak("Okay. We will keep the claim as it is.")

                            stop_audio.set()
                            return

            # -------------------------
            # fallback timer
            # -------------------------

            async def silence_timer():

                await asyncio.sleep(8)

                if not transcript_received.is_set():

                    print("\n⚠️ No response detected. Assuming YES.")

                    self._last_intent = "YES"

                    await self.speak(
                        "We did not hear a response. We will proceed with correcting the claim."
                    )

                    self.send_to_agent4(claim)

                    stop_audio.set()

            await asyncio.gather(
                send_audio(),
                receive_transcript(),
                silence_timer()
            )

            stream.stop_stream()
            stream.close()
            p.terminate()

    # ---------------------------------------------------
    # Main runner
    # ---------------------------------------------------

    async def run(self):

        file_path = "../agent4/agent4_results_20260314_140705.json"

        print("\n📂 Loading claims file...")

        with open(file_path) as f:
            raw = json.load(f)

        claims = raw["results"]

        rejected = [c for c in claims if c["decision"] == "REJECTED"]

        print("📊 Rejected claims:", len(rejected))

        for claim in rejected:

            claim_id = claim.get("claim_id")

            reason = claim.get("reasoning", "")[:200]

            intro = f"""
Hello. I am calling regarding claim {claim_id}.
It was rejected because {reason}.
Would you like us to correct it?
"""

            print("\n📞 Starting call for claim:", claim_id)

            await self.speak(intro)

            await self.run_flux(claim)

            print("\n  Finished claim:", claim_id)

        print("\n🏁 All rejected claims processed.")


if __name__ == "__main__":

    agent = VoiceAgent()

    asyncio.run(agent.run())