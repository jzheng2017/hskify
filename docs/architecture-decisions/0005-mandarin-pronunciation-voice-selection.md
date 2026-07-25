# ADR 0005: Mandarin pronunciation and local voice selection

- Status: Accepted
- Date: 2026-07-25

## Context

Hskify's selection popover can read displayed Simplified Chinese aloud. The
feature is a learning aid, not a text-to-speech service, and it must preserve
the browser companion's local-first privacy boundary.

Firefox exposes voices asynchronously through the Web Speech API. Available
voices, locale tags, quality labels, and linguistic behavior vary by operating
system and Firefox profile. A generic default voice may speak Chinese text with
the wrong language rules, while a remote fallback would introduce undisclosed
text egress and another credential/network dependency.

## Decision

Hskify uses `window.speechSynthesis` and
`SpeechSynthesisUtterance` in the Firefox content layer. It speaks the selected
Chinese characters, not the displayed pinyin.

Voice selection is deterministic for the voices reported by the browser:

1. Reject every voice whose `localService` flag is false.
2. Normalize `_` to `-`, lowercase the BCP 47 tag, and use
   `Intl.Locale.maximize()` when possible.
3. Accept only a Chinese (`zh`) or Mandarin (`cmn`) voice whose maximized script
   is Simplified Han (`Hans`) and region is mainland China (`CN`). A conservative
   tag-pattern fallback is used when locale parsing fails.
4. Prefer exact `zh-CN` or `cmn-CN` tags, then other matching `Hans-CN` voices.
5. Within the matching set, add preference for voice names or URIs containing
   `natural`/`neural`, `premium`, `enhanced`, or an explicit Mandarin marker.
   The browser's default flag is only a small tie-break preference.
6. Preserve browser enumeration order when scores tie.

Firefox may initially return an empty voice list. Hskify listens for
`voiceschanged` and waits for at most two seconds before reporting the voice as
unavailable. Playback uses the selected voice language, a rate of `0.94`,
pitch `1`, and volume `1`.

Activating the same control again stops playback. Opening a different lookup or
dismissing the popover also cancels Hskify-owned playback. Loading, speaking,
unavailable, error, and idle states are exposed through the button label and
ARIA state.

There is no cloud TTS fallback, bundled voice, audio cache, or automatic
download. If no matching local voice exists, the UI tells the user to install
or enable a local Simplified Chinese voice and restart Firefox.

## Consequences

- Selected text stays on the user's machine and no speech credential or service
  configuration is required.
- Playback quality and exact accent depend on the voices exposed by Firefox and
  the operating system.
- A high-quality-sounding name is only a ranking hint. Hskify does not measure
  voice naturalness or guarantee that labels such as "neural" are accurate.
- Traditional-Chinese, Taiwan, Hong Kong, Cantonese, generic non-Simplified
  Chinese, and remote voices are intentionally not fallback candidates.
- Users cannot currently choose among matching voices or change rate, pitch, or
  volume.
- The feature cannot guarantee the correct contextual reading of polyphonic
  characters, tone sandhi, neutral tones, erhua, proper names, loanwords,
  numbers, or punctuation. Those decisions belong to the installed speech
  engine, not `hsk-control` or the pinyin displayed by Hskify.
- The spoken phrase may therefore disagree with dictionary pinyin or a
  pedagogically preferred pronunciation. The text and pinyin remain available
  when speech is missing or inaccurate.
- Browser or OS autoplay policy, disabled speech services, profile state, and
  voice-loading failures can make playback unavailable even when a suitable
  system voice appears to be installed.

## Alternatives considered

### Speak pinyin text

Rejected because general speech engines may pronounce Latin pinyin as English
or spell it out. It also discards the speech engine's Chinese lexical context.

### Accept any Chinese voice

Rejected because Cantonese or Traditional-Chinese regional voices can produce
unexpected readings for Hskify's Simplified-Chinese learning target.

### Use a remote TTS provider

Rejected for the current architecture because it would send selected text off
device, require network and credential behavior, and enlarge the browser
companion's security and disclosure surface.

### Bundle a local neural voice

Deferred because voice models materially increase package size, runtime
complexity, licences, and memory use. Adoption would require a model benchmark,
licence/redistribution audit, resource limits, and a new ADR.

## Verification

Unit tests should cover locale normalization, local-only filtering, scoring and
tie behavior, asynchronous voice arrival, timeout, cancellation ownership,
utterance parameters, and error states. Release smoke testing must use real
Firefox profiles on supported operating systems because mocked Web Speech
objects cannot establish which voices Firefox exposes or how they sound.

Related documents:

- [Architecture overview](../architecture.md)
- [Firefox manual test checklist](../firefox-manual-test-checklist.md)
