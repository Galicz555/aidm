// Browser-side WebRTC handshake against the orchestrator's /api/offer
// endpoint.
//
// Flow:
//   1. getUserMedia (mic, with EC/NS/AGC constraints — those are the
//      single biggest audio-quality lever and the browser does it for
//      free).
//   2. RTCPeerConnection + add mic track + receive transceiver.
//   3. createOffer → setLocalDescription → wait for ICE gathering
//      `complete`. The orchestrator supports trickle-ICE, but waiting
//      for a complete SDP keeps the client simple and adds at most a
//      few hundred ms on a LAN/STUN-less setup.
//   4. POST /api/offer { sdp, type, requestData: { locale } }.
//   5. setRemoteDescription with the answer.
//   6. ontrack → attach to the audio element so the DM voice plays.
//
// The class owns the peer connection, local stream, and any remote
// tracks. `disconnect()` tears them all down so a re-connect starts
// from a clean state.

export type SessionState =
  | 'idle'
  | 'requesting-mic'
  | 'connecting'
  | 'connected'
  | 'reconnecting'
  | 'ending'
  | 'disconnecting'
  | 'error';

export type DataChannelMessage =
  | { type: 'transcript.user'; text: string }
  | { type: 'transcript.bot_delta'; text: string }
  | { type: 'transcript.bot_end' }
  | { type: 'scene.change'; name: string; description?: string }
  | { type: 'character.change'; slug: string }
  | { type: 'beat.change'; content: string }
  | { type: string; [key: string]: unknown };

export interface SessionCallbacks {
  onStateChange: (state: SessionState) => void;
  onError: (message: string) => void;
  onRemoteTrack: (stream: MediaStream) => void;
  onMessage: (message: DataChannelMessage) => void;
}

export interface StartOptions {
  locale: 'hu' | 'en';
  /** Campaign slug picked by the campaign+run selector. Required
   *  server-side; defaults are only for the local CLI. */
  campaignId: string;
  /** Run slug picked by the campaign+run selector. Required server-side
   *  unless ``RUN_SLUG`` env supplies a default — pass it explicitly so
   *  the picker is the source of truth. */
  runId: string;
}

interface OfferResponse {
  sdp: string;
  type: 'answer';
  pc_id: string;
  session_id?: string;
}

const OFFER_ENDPOINT = '/api/offer';

const MIC_CONSTRAINTS: MediaTrackConstraints = {
  echoCancellation: true,
  noiseSuppression: true,
  autoGainControl: true,
  channelCount: 1
};

export class AidmSession {
  private pc: RTCPeerConnection | null = null;
  private localStream: MediaStream | null = null;
  private state: SessionState = 'idle';
  private pcId: string | null = null;
  private sessionId: string | null = null;
  private lastLocale: 'hu' | 'en' | null = null;
  private lastCampaignId: string | null = null;
  private lastRunId: string | null = null;
  // Stop()-initiated and end()-initiated teardowns shouldn't kick off
  // an auto-reconnect. Set just before teardown(), cleared on next
  // start(). Without this we would reconnect into a session the user
  // told us to close.
  private userInitiatedDisconnect = false;

  constructor(private readonly callbacks: SessionCallbacks) {}

  async start(options: StartOptions): Promise<void> {
    if (
      this.state !== 'idle' &&
      this.state !== 'error' &&
      this.state !== 'reconnecting'
    ) {
      this.callbacks.onError(`cannot start: session is ${this.state}`);
      return;
    }

    this.userInitiatedDisconnect = false;
    this.lastLocale = options.locale;
    this.lastCampaignId = options.campaignId;
    this.lastRunId = options.runId;

    try {
      this.transition('requesting-mic');
      this.localStream = await navigator.mediaDevices.getUserMedia({
        audio: MIC_CONSTRAINTS,
        video: false
      });

      // 'connecting' is the normal label; the reconnect path overrides
      // it in attemptResume() to keep the badge readable.
      if (this.state !== 'reconnecting') this.transition('connecting');
      const pc = this.createPeerConnection();
      this.pc = pc;

      for (const track of this.localStream.getAudioTracks()) {
        pc.addTrack(track, this.localStream);
      }
      // Explicit recv transceiver so the answer comes back with an
      // audio m-line even if Pipecat doesn't add one preemptively.
      pc.addTransceiver('audio', { direction: 'recvonly' });

      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);
      await waitForIceGatheringComplete(pc);

      const resumeId = this.state === 'reconnecting' ? this.sessionId : null;
      const answer = await sendOffer(
        pc.localDescription!,
        options.locale,
        options.campaignId,
        options.runId,
        resumeId
      );
      this.pcId = answer.pc_id;
      if (answer.session_id) this.sessionId = answer.session_id;
      await pc.setRemoteDescription({ type: answer.type, sdp: answer.sdp });

      // The connectionstatechange handler flips us to 'connected' once
      // ICE actually succeeds. createOffer/setRemoteDescription only
      // confirm signaling, not media flow.
    } catch (err) {
      this.handleFailure(err);
    }
  }

  /** Politely end the session: persona writes the log + says goodbye.
   *
   *  The page is expected to call ``stop()`` once it sees the next
   *  ``transcript.bot_end`` data-channel message after this — that's
   *  when the DM has actually finished speaking the farewell. We
   *  don't auto-disconnect here because cutting audio before TTS
   *  finishes is jarring.
   */
  async end(): Promise<void> {
    if (this.state !== 'connected' || !this.pcId) return;
    this.userInitiatedDisconnect = true;
    this.transition('ending');
    try {
      const response = await fetch(`/api/session/${this.pcId}/end`, {
        method: 'POST'
      });
      if (!response.ok) {
        // Server didn't know about this pc_id, or the pipeline was
        // already gone — fall through to a hard stop().
        this.callbacks.onError(`end-session rejected (${response.status})`);
        await this.stop();
      }
    } catch (err) {
      this.callbacks.onError(err instanceof Error ? err.message : String(err));
      await this.stop();
    }
  }

  async stop(): Promise<void> {
    if (this.state === 'idle' || this.state === 'disconnecting') return;
    this.userInitiatedDisconnect = true;
    this.transition('disconnecting');
    this.teardown();
    this.pcId = null;
    this.sessionId = null;
    this.lastLocale = null;
    this.transition('idle');
  }

  /** True while we're awaiting the DM's farewell after ``end()``. */
  get isEnding(): boolean {
    return this.state === 'ending';
  }

  private createPeerConnection(): RTCPeerConnection {
    const pc = new RTCPeerConnection({
      // No STUN/TURN by default — dev is same-host LAN. Production
      // remote use needs at least a public STUN server here, which is
      // a deploy-time decision not a code one.
      iceServers: []
    });

    pc.onconnectionstatechange = () => {
      switch (pc.connectionState) {
        case 'connected':
          this.transition('connected');
          break;
        case 'failed':
          if (this.canResume()) {
            void this.attemptResume();
          } else {
            this.callbacks.onError('WebRTC connection failed');
            this.transition('error');
            this.teardown();
          }
          break;
        case 'closed':
        case 'disconnected':
          if (this.state === 'idle' || this.state === 'disconnecting') break;
          if (this.canResume()) {
            void this.attemptResume();
          } else {
            this.transition('idle');
            this.teardown();
          }
          break;
      }
    };

    pc.ontrack = (event) => {
      const [stream] = event.streams;
      if (stream) this.callbacks.onRemoteTrack(stream);
    };

    // The orchestrator opens the data channel from its side. We listen
    // here for the remote-initiated channel and route every JSON
    // message to onMessage. Non-JSON payloads are dropped silently
    // because the channel is shared with pipecat signaling traffic
    // (e.g. PeerLeftMessage) that we do not interpret.
    pc.ondatachannel = (event) => {
      const channel = event.channel;
      channel.onmessage = (msg) => {
        if (typeof msg.data !== 'string') return;
        try {
          const payload = JSON.parse(msg.data) as DataChannelMessage;
          this.callbacks.onMessage(payload);
        } catch {
          /* non-JSON frame — ignore */
        }
      };
    };

    return pc;
  }

  private canResume(): boolean {
    // Only resume on transport-level drops while we're in a healthy
    // session. End-session / explicit stop set the flag so the
    // farewell teardown doesn't bounce back. State 'reconnecting'
    // means we are already trying once — don't retry-loop.
    return (
      !this.userInitiatedDisconnect &&
      this.state === 'connected' &&
      this.sessionId !== null &&
      this.lastLocale !== null &&
      this.lastCampaignId !== null &&
      this.lastRunId !== null
    );
  }

  private async attemptResume(): Promise<void> {
    const locale = this.lastLocale!;
    const campaignId = this.lastCampaignId ?? '';
    const runId = this.lastRunId ?? '';
    this.transition('reconnecting');
    // Tear down the dead peer; start() builds a fresh one and sends
    // the stored session_id, which the server uses to claim its
    // pending LLMContext (5s grace window — see
    // docs/session-lifecycle.md).
    this.teardownPeerOnly();
    await this.start({ locale, campaignId, runId });
  }

  private handleFailure(err: unknown): void {
    const message = err instanceof Error ? err.message : String(err);
    this.callbacks.onError(message);
    this.transition('error');
    this.teardown();
  }

  private teardown(): void {
    this.teardownPeerOnly();
    if (this.localStream) {
      for (const track of this.localStream.getTracks()) track.stop();
      this.localStream = null;
    }
  }

  private teardownPeerOnly(): void {
    if (this.pc) {
      this.pc.onconnectionstatechange = null;
      this.pc.ontrack = null;
      this.pc.ondatachannel = null;
      this.pc.close();
      this.pc = null;
    }
  }

  private transition(next: SessionState): void {
    this.state = next;
    this.callbacks.onStateChange(next);
  }
}

function waitForIceGatheringComplete(pc: RTCPeerConnection): Promise<void> {
  if (pc.iceGatheringState === 'complete') return Promise.resolve();
  return new Promise((resolve) => {
    const check = () => {
      if (pc.iceGatheringState === 'complete') {
        pc.removeEventListener('icegatheringstatechange', check);
        resolve();
      }
    };
    pc.addEventListener('icegatheringstatechange', check);
  });
}

async function sendOffer(
  localDescription: RTCSessionDescription,
  locale: 'hu' | 'en',
  campaignId: string,
  runId: string,
  resumeSessionId: string | null
): Promise<OfferResponse> {
  const requestData: Record<string, string> = {
    locale,
    campaign_id: campaignId,
    run_id: runId,
  };
  if (resumeSessionId) requestData.session_id = resumeSessionId;
  const response = await fetch(OFFER_ENDPOINT, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      sdp: localDescription.sdp,
      type: localDescription.type,
      requestData
    })
  });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(`offer rejected (${response.status}): ${detail}`);
  }
  return (await response.json()) as OfferResponse;
}
