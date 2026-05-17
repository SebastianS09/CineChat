"""Mic → OggOpus → /ws/phone WebSocket test.

Uses ffmpeg subprocess pipes for codec work so gradbot gets proper OggOpus
timestamps (self-pacing) instead of raw PCM byte counts.

Flow:
  encode pipe: sounddevice mic (48kHz int16) → ffmpeg stdin → OggOpus stdout → WS binary
  decode pipe: WS binary → ffmpeg stdin → PCM 48kHz stdout → sounddevice speakers

Usage:
    python test_ws_mic.py [--url ws://localhost:8000/ws/phone] [--lang en]
"""

import argparse
import asyncio
import json
import sys

import numpy as np
import sounddevice as sd
import websockets

DEVICE_RATE  = 48000
CHANNELS     = 1
CHUNK_SAMPLES = int(DEVICE_RATE * 0.020)   # 20 ms
CHUNK_BYTES   = CHUNK_SAMPLES * 2           # int16


# ── ffmpeg pipe launchers ──────────────────────────────────────────────────────

async def start_encode_pipe() -> asyncio.subprocess.Process:
    """mic PCM (s16le 48kHz) → OggOpus via ffmpeg."""
    return await asyncio.create_subprocess_exec(
        "ffmpeg", "-loglevel", "error",
        "-f", "s16le", "-ar", str(DEVICE_RATE), "-ac", "1", "-i", "pipe:0",
        "-f", "ogg", "-acodec", "libopus", "-ar", "48000", "-ac", "1",
        "-frame_duration", "20",
        "pipe:1",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )


async def start_decode_pipe() -> asyncio.subprocess.Process:
    """OggOpus → PCM s16le 48kHz via ffmpeg."""
    return await asyncio.create_subprocess_exec(
        "ffmpeg", "-loglevel", "error",
        "-f", "ogg", "-i", "pipe:0",
        "-f", "s16le", "-ar", str(DEVICE_RATE), "-ac", "1",
        "pipe:1",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )


# ── Speaker playback via asyncio ───────────────────────────────────────────────

class Speaker:
    """Reads decoded PCM from ffmpeg stdout and plays it via sounddevice."""

    def __init__(self):
        self._buf = bytearray()
        self._lock = asyncio.Lock()
        self._stream: sd.OutputStream | None = None

    def start(self):
        self._stream = sd.OutputStream(
            samplerate=DEVICE_RATE, channels=CHANNELS, dtype="int16",
            blocksize=CHUNK_SAMPLES, callback=self._cb,
        )
        self._stream.start()

    def _cb(self, outdata, frames, time, status):
        need = frames * 2
        chunk = bytes(self._buf[:need])
        del self._buf[:need]
        if len(chunk) < need:
            chunk = chunk + bytes(need - len(chunk))
        outdata[:, 0] = np.frombuffer(chunk, dtype=np.int16)

    def feed(self, pcm_bytes: bytes):
        self._buf.extend(pcm_bytes)

    def stop(self):
        if self._stream:
            self._stream.stop()
            self._stream.close()


# ── Main ───────────────────────────────────────────────────────────────────────

async def run(url: str, lang: str):
    print(f"Connecting to {url} ...")
    speaker = Speaker()
    speaker.start()

    enc_proc = await start_encode_pipe()
    dec_proc = await start_decode_pipe()

    done = asyncio.Event()
    bot_talking = False

    async with websockets.connect(url) as ws:
        # Send start handshake (gradbot expects {"type": "start", ...})
        await ws.send(json.dumps({"type": "start", "language": lang}))
        print("Connected. Speak into your mic. Ctrl+C to hang up.\n")

        # ── mic → encode pipe → WS ──────────────────────────────────────────
        mic_q: asyncio.Queue[bytes] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def mic_cb(indata, frames, time, status):
            loop.call_soon_threadsafe(mic_q.put_nowait, indata[:, 0].copy().astype(np.int16).tobytes())

        mic_stream = sd.InputStream(
            samplerate=DEVICE_RATE, channels=CHANNELS, dtype="int16",
            blocksize=CHUNK_SAMPLES, callback=mic_cb,
        )

        async def send_loop():
            mic_stream.start()
            try:
                while not done.is_set():
                    pcm = await asyncio.wait_for(mic_q.get(), timeout=1.0)
                    if bot_talking:
                        continue
                    enc_proc.stdin.write(pcm)
                    await enc_proc.stdin.drain()
            except asyncio.TimeoutError:
                pass
            except Exception as e:
                print(f"[send_loop] {type(e).__name__}: {e}", file=sys.stderr)
            finally:
                mic_stream.stop()
                try:
                    enc_proc.stdin.close()
                except Exception:
                    pass

        async def encode_to_ws_loop():
            """Read OggOpus chunks from ffmpeg stdout and send as binary WS frames."""
            try:
                while not done.is_set():
                    chunk = await enc_proc.stdout.read(4096)
                    if not chunk:
                        break
                    await ws.send(chunk)
            except Exception as e:
                print(f"[encode_to_ws] {type(e).__name__}: {e}", file=sys.stderr)

        async def recv_loop():
            nonlocal bot_talking
            try:
                async for msg in ws:
                    if isinstance(msg, bytes):
                        # OggOpus audio from bot — pipe into decoder
                        dec_proc.stdin.write(msg)
                        await dec_proc.stdin.drain()
                    else:
                        try:
                            data = json.loads(msg)
                        except Exception:
                            continue
                        event = data.get("event") or data.get("type")
                        if event == "bot_start":
                            bot_talking = True
                            print("[mic] MUTED — bot speaking", file=sys.stderr)
                        elif event == "bot_stop":
                            bot_talking = False
                            print("[mic] UNMUTED", file=sys.stderr)
                        elif event == "transcript":
                            print(f"\n[STT] {data.get('text', '')}")
                        else:
                            print(f"[SERVER] {event}: {json.dumps(data)[:120]}", file=sys.stderr)
            except Exception as e:
                print(f"[recv_loop] {type(e).__name__}: {e}", file=sys.stderr)
            finally:
                done.set()

        async def decode_to_speaker_loop():
            """Read decoded PCM from ffmpeg and push to speaker."""
            try:
                while not done.is_set():
                    pcm = await dec_proc.stdout.read(CHUNK_BYTES)
                    if not pcm:
                        break
                    speaker.feed(pcm)
            except Exception as e:
                print(f"[decode_to_speaker] {type(e).__name__}: {e}", file=sys.stderr)

        try:
            await asyncio.gather(
                send_loop(),
                encode_to_ws_loop(),
                recv_loop(),
                decode_to_speaker_loop(),
            )
        except (KeyboardInterrupt, asyncio.CancelledError):
            done.set()

    # Cleanup
    speaker.stop()
    for proc in (enc_proc, dec_proc):
        try:
            proc.terminate()
            await asyncio.wait_for(proc.wait(), timeout=2.0)
        except Exception:
            pass
    print("\nCall ended.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url",  default="ws://localhost:8000/ws/phone")
    parser.add_argument("--lang", default="en")
    args = parser.parse_args()

    try:
        asyncio.run(run(args.url, args.lang))
    except KeyboardInterrupt:
        pass
