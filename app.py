"""LectureLens - Streamlit UI.  Run:  streamlit run app.py"""
import html
import json

import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="LectureLens", page_icon="🎓", layout="wide")

import config  # noqa: E402
from pipeline import export, ingest, llm, notes as notes_mod, profile as profile_mod, quiz as quiz_mod  # noqa: E402
from pipeline import rag, store  # noqa: E402
from pipeline.diagrams import ink_url  # noqa: E402
from pipeline.flashcards import to_anki_csv  # noqa: E402
from pipeline.run import OUTPUTS, KitOptions, build_kit, list_kits, load_kit, prepare_lecture  # noqa: E402
from pipeline.schemas import LearnerProfile  # noqa: E402
from pipeline.utils import fmt_ts, parse_ts  # noqa: E402

ss = st.session_state
for name, value in {"lecture_id": None, "profile": None, "interview": [], "kit": None, "seek": 0,
                    "seek_nonce": 0, "chat": [], "followups": {}, "compare": None}.items():
    ss.setdefault(name, value)

LANGUAGES = {"auto": "Auto-detect", "en": "English", "hi": "Hindi / Hinglish"}
OUTPUT_LABELS = {"notes": "Notes", "quiz": "Quiz", "flashcards": "Flashcards", "diagrams": "Diagrams",
                 "audio": "Audio recap", "faithfulness": "Faithfulness check"}


# ------------------------------------------------------------------ helpers
def seek_to(seconds):
    ss.seek = int(seconds or 0)
    ss.seek_nonce += 1


def ts_button(ts: str, key: str):
    """Small button that jumps the player (in the sidebar) to a timestamp."""
    if not ts:
        return
    kwargs = dict(key=key, on_click=seek_to, args=(parse_ts(ts),), help="Jump the player to this moment")
    try:
        st.button(f"▶ {ts}", type="tertiary", **kwargs)
    except Exception:                       # older Streamlit without the tertiary type
        st.button(f"▶ {ts}", **kwargs)


def render_player(meta):
    kind, path, start = meta.media_kind, meta.media_path, int(ss.seek)
    if kind == "none" or not path:
        st.caption("No media for this lecture (transcript only).")
        return
    player = st.audio if kind == "audio" else st.video
    try:
        player(path, start_time=start, autoplay=ss.seek_nonce > 0)
    except TypeError:
        player(path, start_time=start)
    st.caption(f"Position: {fmt_ts(start)} · click any ▶ timestamp to jump")


def mermaid(code: str, height: int = 460):
    components.html(f"""
<div class="mermaid" style="display:flex;justify-content:center">{html.escape(code)}</div>
<script type="module">
  import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10.9.1/dist/mermaid.esm.min.mjs';
  mermaid.initialize({{startOnLoad: true, theme: 'neutral', securityLevel: 'strict'}});
</script>""", height=height, scrolling=True)


def flashcard_grid(cards):
    items = "".join(
        f"""<div class="card" onclick="this.classList.toggle('flip')"><div class="inner">
        <div class="face front"><span class="tag">{html.escape(c.kind)}</span>{html.escape(c.front)}</div>
        <div class="face back">{html.escape(c.back)}<small>{html.escape(c.ts)}</small></div></div></div>"""
        for c in cards)
    rows = (len(cards) + 2) // 3
    components.html(f"""
<style>
 body{{font-family:system-ui,sans-serif;margin:0}}
 .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:14px;padding:4px}}
 .card{{height:170px;perspective:900px;cursor:pointer}}
 .inner{{position:relative;width:100%;height:100%;transition:transform .5s;transform-style:preserve-3d}}
 .card.flip .inner{{transform:rotateY(180deg)}}
 .face{{position:absolute;inset:0;backface-visibility:hidden;border-radius:12px;padding:16px;
        display:flex;flex-direction:column;justify-content:center;font-size:15px;line-height:1.35;
        box-shadow:0 1px 4px rgba(0,0,0,.15);overflow:auto}}
 .front{{background:#f4f6fb;color:#1d2433}}
 .back{{background:#1f4e8c;color:#fff;transform:rotateY(180deg)}}
 .tag{{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:#6b7590;margin-bottom:8px}}
 small{{margin-top:10px;opacity:.75}}
</style><div class="grid">{items}</div>""", height=min(1600, rows * 190 + 20), scrolling=True)


def profile_card(profile: LearnerProfile):
    weak = ", ".join(profile.weak_areas) or "—"
    st.markdown(f"**Level:** {profile.level} · **Goal:** {profile.goal} · **Depth:** {profile.depth} · "
                f"**Language:** {profile.language} · **Quiz size:** {profile.quiz_size}  \n**Weak areas:** {weak}")


def run_with_progress(fn, *args, **kwargs):
    bar = st.progress(0.0, text="Starting…")

    def progress(label, fraction):
        bar.progress(min(1.0, max(0.0, fraction)), text=label)

    try:
        return fn(*args, progress=progress, **kwargs)
    finally:
        bar.empty()


# ------------------------------------------------------------------ sidebar: lecture + settings
providers = llm.available_providers()
lectures = store.list_lectures()
meta = store.load_meta(ss.lecture_id) if ss.lecture_id else None

with st.sidebar:
    st.title("🎓 LectureLens")
    st.caption("Recorded lecture → personalised study kit")
    if meta:
        st.markdown(f"**{meta.title}**")
        render_player(meta)

    if not providers:
        st.error("No LLM API key found. Copy `.env.example` to `.env` and add GEMINI_API_KEY and/or GROQ_API_KEY.")
    else:
        st.caption("LLM APIs: " + " · ".join(f"{p} ✓" for p in providers)
                   + ("" if config.GROQ_API_KEY else " · Whisper API ✗ (local fallback)"))

    with st.expander("📥 Lecture", expanded=meta is None):
        if lectures:
            labels = {m.id: f"{m.title} ({fmt_ts(m.duration)})" for m in lectures}
            options = [None] + list(labels)
            choice = st.selectbox("Open a processed lecture", options,
                                  index=options.index(ss.lecture_id) if ss.lecture_id in options else 0,
                                  format_func=lambda i: "—" if i is None else labels[i])
            if choice != ss.lecture_id:
                ss.lecture_id, ss.kit, ss.chat, ss.seek, ss.followups, ss.compare = choice, None, [], 0, {}, None
                st.rerun()
            st.divider()

        mode = st.radio("Add a new lecture", ["Upload video/audio", "YouTube link", "Transcript file"])
        language = st.selectbox("Lecture language", list(LANGUAGES), format_func=LANGUAGES.get)
        new_meta = None
        try:
            if mode == "Upload video/audio":
                up = st.file_uploader("Video or audio", type=[e[1:] for e in ingest.VIDEO_EXT | ingest.AUDIO_EXT])
                title = st.text_input("Title (optional)")
                if up and st.button("Process lecture", type="primary"):
                    with st.spinner("Converting audio…"):
                        new_meta = ingest.ingest_file(up.getvalue(), up.name, title or None, language)
            elif mode == "YouTube link":
                url = st.text_input("YouTube URL")
                if url and st.button("Process lecture", type="primary"):
                    with st.spinner("Downloading audio…"):
                        new_meta = ingest.ingest_youtube(url.strip(), language)
            else:
                up = st.file_uploader("Transcript (.srt / .vtt / .txt)", type=["srt", "vtt", "txt"])
                title = st.text_input("Title (optional)")
                media = st.text_input("YouTube URL for the player (optional)")
                if up and st.button("Process lecture", type="primary"):
                    new_meta = ingest.ingest_transcript(up.getvalue(), up.name, title or None, language, media.strip())
            if new_meta:
                run_with_progress(prepare_lecture, new_meta.id,
                                  language=None if language == "auto" else language)
                ss.lecture_id, ss.kit, ss.chat, ss.seek, ss.compare = new_meta.id, None, [], 0, None
                st.rerun()
        except Exception as exc:
            st.error(f"Processing failed: {exc}")

    with st.expander("⚙️ Generation settings", expanded=False):
        provider = st.selectbox("Generator LLM", providers or ["gemini"],
                                index=(providers.index(config.LLM_PROVIDER) if config.LLM_PROVIDER in providers else 0))
        model = st.text_input("Model", value=llm.default_model(provider))
        strategy = st.selectbox("Prompting strategy", list(notes_mod.STRATEGIES),
                                index=list(notes_mod.STRATEGIES).index(config.DEFAULT_STRATEGY),
                                format_func=notes_mod.STRATEGIES.get,
                                help="Switch live to compare zero-shot, few-shot, CoT and ToT notes.")
        temperature = st.slider("Temperature (notes)", 0.0, 1.2, config.NOTES_TEMPERATURE, 0.05,
                                help="Low for factual notes. Tree-of-Thought candidates always use 0.8.")
        outputs = st.multiselect("Outputs", list(OUTPUTS), default=list(OUTPUTS), format_func=OUTPUT_LABELS.get)
        podcast = st.toggle("Two-voice podcast recap", value=False)
        quiz_cot = st.toggle("Chain-of-Thought quiz", value=True)
        try:
            judge = llm.resolve("judge")
            st.caption(f"Judge model: {judge[0]}/{judge[1]}")
        except llm.LLMError:
            pass

options = KitOptions(strategy=strategy, provider=provider, model=model, temperature=temperature,
                     outputs=tuple(outputs) or ("notes",), podcast=podcast, quiz_cot=quiz_cot)


# ------------------------------------------------------------------ landing page
if meta is None:
    st.title("LectureLens")
    st.subheader("Turn a recorded lecture into a personalised, grounded study kit")
    c1, c2, c3 = st.columns(3)
    c1.markdown("**1 · Add a lecture**  \nUpload a video or audio file, paste a YouTube link, or use a transcript "
                "(sidebar). English and Hindi/Hinglish are supported.")
    c2.markdown("**2 · A short interview**  \nThree to five questions about your level, goal and weak areas "
                "personalise everything that follows.")
    c3.markdown("**3 · Your study kit**  \nTimestamped notes, diagrams, an adaptive quiz, flashcards, an audio "
                "recap, a lecture chatbot with citations, and a faithfulness report.")
    st.info("Whisper → topic-aware chunking → Tree-of-Thought notes, Chain-of-Thought quiz, Mermaid diagrams, "
            "TTS recap, RAG chatbot → claim-level faithfulness check.")
    st.stop()

chunks = store.load_chunks(meta.id) or []
st.title(meta.title)
st.caption(f"{fmt_ts(meta.duration)} · {LANGUAGES.get(meta.language, meta.language)} · {len(chunks)} topic chunks · "
           f"source: {meta.source_type}")


# ------------------------------------------------------------------ step 1: interview
with st.container(border=True):
    if ss.profile is None:
        st.subheader("Step 1 · Quick interview")
        left, right = st.columns([3, 2])
        with right:
            st.markdown("**…or skip with a preset**")
            preset = st.selectbox("Preset profile", list(profile_mod.PRESETS), label_visibility="collapsed")
            if st.button("Use this profile"):
                ss.profile = profile_mod.PRESETS[preset].model_copy(deep=True)
                st.rerun()
        with left:
            for msg in ss.interview:
                st.chat_message(msg["role"]).write(msg["content"])
            if not ss.interview:
                if st.button("Start the interview", type="primary", disabled=not providers):
                    try:
                        turn = profile_mod.next_turn([], meta.title, [c.title for c in chunks])
                        ss.interview.append({"role": "assistant", "content": turn.question})
                    except llm.LLMError as exc:
                        st.error(str(exc))
                    st.rerun()
            else:
                with st.form("interview_form", clear_on_submit=True):
                    reply = st.text_input("Your answer")
                    sent = st.form_submit_button("Send")
                if sent and reply.strip():
                    ss.interview.append({"role": "user", "content": reply.strip()})
                    try:
                        turn = profile_mod.next_turn(ss.interview, meta.title, [c.title for c in chunks])
                    except llm.LLMError as exc:
                        st.error(str(exc))
                        st.stop()
                    if turn.done and turn.profile:
                        ss.profile = turn.profile
                    else:
                        ss.interview.append({"role": "assistant", "content": turn.question})
                    st.rerun()
    else:
        head, edit = st.columns([5, 1])
        head.markdown("**Learner profile** (injected into every prompt)")
        if edit.button("Redo interview"):
            ss.profile, ss.interview, ss.kit = None, [], None
            st.rerun()
        profile_card(ss.profile)


# ------------------------------------------------------------------ step 2: generate / load kit
if ss.profile is not None:
    saved = list_kits(meta.id)
    gen_col, load_col = st.columns([2, 3])
    with gen_col:
        if st.button("✨ Generate study kit", type="primary", disabled=not providers, use_container_width=True):
            try:
                ss.kit = run_with_progress(build_kit, meta.id, ss.profile, options)
                ss.followups = {}
            except Exception as exc:
                st.error(f"Generation failed: {exc}")
    with load_col:
        if saved:
            labels = {m["key"]: f"{notes_mod.STRATEGIES.get(m['options']['strategy'], '?')} · "
                                f"{m['profile']['level']} / {m['profile']['goal']} · {m['created'][5:16]}"
                      for m in saved}
            pick = st.selectbox("Saved study kits (load instantly, no API calls)", [None] + list(labels),
                                format_func=lambda k: "—" if k is None else labels[k])
            if pick and (ss.kit is None or ss.kit["key"] != pick):
                ss.kit = load_kit(meta.id, pick)
                ss.profile = ss.kit["profile"]
                ss.followups = {}
                st.rerun()

kit = ss.kit
if kit is None or kit["meta"].id != meta.id:
    st.stop()

youtube_url = meta.media_path if meta.media_kind == "youtube" else None
claims_by_section: dict[int, list] = {}
if kit.get("faithfulness"):
    for claim in kit["faithfulness"].claims:
        if claim.verdict != "supported":
            claims_by_section.setdefault(claim.section, []).append(claim)

tabs = st.tabs(["📝 Notes", "🗺️ Diagrams", "❓ Quiz", "🃏 Flashcards", "🎧 Audio recap", "💬 Ask the Lecture",
                "✅ Quality report", "👥 Compare profiles", "📜 Transcript"])

# ------------------------------------------------------------------ notes
with tabs[0]:
    notes = kit["notes"]
    st.caption(f"Strategy: {notes_mod.STRATEGIES.get(notes.strategy, notes.strategy)}")
    if notes.summary:
        st.info("\n".join(f"- {s}" for s in notes.summary))
    for i, section in enumerate(notes.sections):
        head, btn = st.columns([6, 1])
        head.subheader(section.title)
        with btn:
            ts_button(section.ts, f"sec{i}")
        for j, bullet in enumerate(section.bullets):
            b1, b2 = st.columns([1, 8])
            with b1:
                ts_button(bullet.ts, f"b{i}-{j}")
            b2.markdown(bullet.text)
        if section.definitions:
            st.markdown("**Definitions**  \n" + "  \n".join(
                f"**{d.term}** — {d.definition} `{d.ts}`" for d in section.definitions))
        if section.formulas:
            st.markdown("**Formulas**")
            for f in section.formulas:
                st.code(f.expression, language=None)
                if f.meaning:
                    st.caption(f"{f.meaning} · {f.ts}")
        if section.examples:
            st.markdown("**Examples**  \n" + "  \n".join(f"• {e}" for e in section.examples))
        for claim in claims_by_section.get(i, []):
            st.warning(f"⚠️ {claim.verdict.replace('_', ' ').title()}: “{claim.claim}” — {claim.reason}")
        st.divider()
    if notes.glossary:
        with st.expander("Key terms", expanded=True):
            for g in notes.glossary:
                st.markdown(f"**{g.term}** — {g.definition}")
    md = export.notes_to_markdown(notes, youtube_url)
    d1, d2 = st.columns(2)
    d1.download_button("⬇ Notes (Markdown)", md, f"{meta.title}_notes.md", "text/markdown")
    try:
        d2.download_button("⬇ Notes (PDF)", export.markdown_to_pdf(md), f"{meta.title}_notes.pdf", "application/pdf")
    except Exception as exc:
        d2.caption(f"PDF export unavailable: {exc}")

# ------------------------------------------------------------------ diagrams
with tabs[1]:
    diagrams = kit.get("diagrams")
    if not diagrams:
        st.caption("Diagrams were not generated for this kit.")
    else:
        stats = kit.get("diagram_stats") or {}
        if stats.get("planned"):
            st.caption(f"First-try valid Mermaid: {stats['first_try_valid']}/{stats['planned']} · "
                       f"valid after error-feedback retries: {stats['valid_after_retries']}/{stats['planned']}")
        for k, d in enumerate(diagrams):
            head, btn = st.columns([6, 1])
            head.markdown(f"#### {d.title}  \n_{d.kind}{' · ' + d.concept if d.concept else ''}_")
            with btn:
                ts_button(d.ts, f"dg{k}")
            mermaid(d.mermaid, height=520 if d.kind == "mindmap" else 440)
            with st.expander("Mermaid code"):
                st.code(d.mermaid, language=None)
                st.markdown(f"[Open as SVG]({ink_url(d.mermaid)}) · attempts: {d.attempts}"
                            + (f" · note: {d.error}" if d.error else ""))

# ------------------------------------------------------------------ quiz
with tabs[2]:
    quiz = kit.get("quiz")
    if not quiz or not quiz.questions:
        st.caption("No quiz for this kit.")
    else:
        stats = quiz.stats
        if stats.get("generated"):
            st.caption(f"{len(quiz.questions)} questions · self-verification: {stats['passed_first_try']}/"
                       f"{stats['generated']} passed first time, {stats['regenerated']} regenerated, "
                       f"{stats['dropped']} dropped · {'CoT' if stats.get('use_cot') else 'no CoT'}")
        correct = answered = 0

        def ask_question(q, key):
            choice = st.radio(q.stem, list(range(4)), index=None, key=key,
                              format_func=lambda j: f"{'ABCD'[j]}) {q.options[j]}")
            if choice is None:
                return None
            if choice == q.answer_index:
                st.success("Correct! " + q.explanation)
            else:
                st.error(f"Not quite — the answer is {'ABCD'[q.answer_index]}. {q.explanation}")
            return choice

        for i, q in enumerate(quiz.questions):
            with st.container(border=True):
                st.caption(f"Q{i + 1} · {q.bloom} · {q.difficulty} · concept: {q.concept}")
                choice = ask_question(q, f"q-{kit['key']}-{i}")
                if choice is not None:
                    answered += 1
                    correct += choice == q.answer_index
                    ts_button(q.ts, f"qts{i}")
                    if choice != q.answer_index:
                        fkey = f"{kit['key']}-{i}"
                        if fkey not in ss.followups:
                            if st.button("Try an easier question on this concept", key=f"fu{i}"):
                                with st.spinner("Writing a follow-up…"):
                                    ss.followups[fkey] = quiz_mod.generate_followup(q, choice, kit["chunks"], kit["profile"])
                                st.rerun()
                        else:
                            st.markdown("**Follow-up**")
                            ask_question(ss.followups[fkey], f"fq-{fkey}")
                    if q.reasoning:
                        with st.expander("How this question was built (CoT)"):
                            st.write(q.reasoning)
        if answered:
            st.metric("Score", f"{correct}/{answered}", help="Follow-up questions are not counted.")
        st.download_button("⬇ Quiz (Markdown)", export.quiz_to_markdown(quiz, f"{meta.title} — quiz"),
                           f"{meta.title}_quiz.md", "text/markdown")
        try:
            st.download_button("⬇ Quiz (PDF)", export.markdown_to_pdf(export.quiz_to_markdown(quiz, f"{meta.title} — quiz")),
                               f"{meta.title}_quiz.pdf", "application/pdf")
        except Exception as exc:
            st.caption(f"PDF export unavailable: {exc}")

# ------------------------------------------------------------------ flashcards
with tabs[3]:
    cards = kit.get("flashcards")
    if not cards or not cards.cards:
        st.caption("No flashcards for this kit.")
    else:
        st.caption(f"{len(cards.cards)} cards · click a card to flip it")
        flashcard_grid(cards.cards)
        st.download_button("⬇ Anki CSV", to_anki_csv(cards.cards, meta.title), f"{meta.title}_flashcards.csv",
                           "text/csv", help="Anki → File → Import → choose 'Basic' note type, comma separated.")

# ------------------------------------------------------------------ audio recap
with tabs[4]:
    recap = kit.get("recap")
    if not recap:
        st.caption("No audio recap for this kit.")
    else:
        if kit.get("recap_audio"):
            st.audio(str(kit["recap_audio"]))
            st.download_button("⬇ MP3", kit["recap_audio"].read_bytes(), f"{meta.title}_recap.mp3", "audio/mpeg")
        else:
            st.warning("Script written, but speech synthesis failed (edge-tts needs internet).")
        with st.expander("Script", expanded=False):
            for line in recap.lines:
                st.markdown(f"**{line.speaker.title()}:** {line.text}")

# ------------------------------------------------------------------ ask the lecture
with tabs[5]:
    titles = {m.id: m.title for m in lectures}
    scope = st.multiselect("Search in", list(titles), default=[meta.id], format_func=titles.get,
                           help="Pick several lectures to ask across a whole course.")
    for n, msg in enumerate(ss.chat):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                for c in msg.get("citations", []):
                    if c["valid"] and c["lecture_id"] == meta.id:
                        ts_button(c["ts"], f"cite{n}-{c['ts']}")
                if msg.get("hits"):
                    with st.expander("Retrieved excerpts"):
                        for h in msg["hits"]:
                            st.markdown(f"**{h['lecture_title']} · {fmt_ts(h['start'])}–{fmt_ts(h['end'])} · "
                                        f"{h['title']}** (similarity {h['score']:.2f})")
                            st.caption(h["text"][:600] + "…")
    question = st.chat_input("Ask a question about the lecture", key="ask_input")
    if question and scope:
        history = [{"role": m["role"], "content": m["content"]} for m in ss.chat]
        ss.chat.append({"role": "user", "content": question})
        try:
            with st.spinner("Searching the lecture…"):
                result = rag.ask(question, scope, kit["profile"], history)
            content = result["answer"] if result["found"] else f"🤷 {result['answer']}"
            ss.chat.append({"role": "assistant", "content": content, "citations": result["citations"],
                            "hits": result["hits"]})
        except llm.LLMError as exc:
            ss.chat.append({"role": "assistant", "content": f"Error: {exc}"})
        st.rerun()

# ------------------------------------------------------------------ quality report
with tabs[6]:
    report = kit.get("faithfulness")
    if report and report.total:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Faithfulness", f"{report.score:.0%}")
        c2.metric("Supported", report.supported)
        c3.metric("Not mentioned", report.not_mentioned)
        c4.metric("Contradicted", report.contradicted)
        st.caption(f"{report.total} atomic claims checked against the transcript by the judge model.")
        show = st.radio("Show", ["Unsupported only", "All claims"], horizontal=True)
        badge = {"supported": "🟢", "not_mentioned": "🟡", "contradicted": "🔴"}
        for n, c in enumerate(report.claims):
            if show == "Unsupported only" and c.verdict == "supported":
                continue
            where = "Summary" if c.section < 0 else kit["notes"].sections[c.section].title
            st.markdown(f"{badge.get(c.verdict, '⚪')} **{c.claim}**  \n<small>{where} · {c.reason}</small>",
                        unsafe_allow_html=True)
    else:
        st.caption("Faithfulness check was not run for this kit.")

    st.subheader("Cost of this kit")
    cost = kit.get("cost") or {}
    if cost:
        st.dataframe([{"stage": k, **v} for k, v in cost.items()], use_container_width=True, hide_index=True)
    else:
        st.caption("Loaded from cache — no new API calls.")

    log = kit.get("notes_log") or {}
    trees = [c for c in log.get("chunks", []) if "levels" in c]
    if trees:
        st.subheader("Tree-of-Thought search")
        tree = st.selectbox("Chunk", trees, format_func=lambda t: f"Chunk {t['chunk_id'] + 1}: {t['title']}")
        st.caption(f"breadth {tree['breadth']} · keep {tree['keep']} · depth {tree['depth']} · {tree['calls']} LLM calls · "
                   f"selected node {tree['selected']}")
        for level in tree["levels"]:
            st.markdown(f"**Depth {level['depth']}**")
            cols = st.columns(len(level["nodes"]))
            for col, node in zip(cols, level["nodes"]):
                with col.container(border=True):
                    mark = "✅ kept" if node.get("kept") else "✂️ pruned"
                    st.markdown(f"**{node['id']}** · {node['style']}  \n{mark} · total **{node['score'].get('total')}**")
                    s = node["score"]
                    st.caption(f"coverage {s.get('coverage')} · faithfulness {s.get('faithfulness')} · "
                               f"clarity {s.get('clarity')} · fit {s.get('profile_fit')}")
                    st.caption(s.get("reason", ""))
                    with st.expander("Notes"):
                        st.json(node["section"], expanded=False)
    cot = [c for c in log.get("chunks", []) if "reasoning" in c]
    if cot:
        st.subheader("Chain-of-Thought reasoning")
        for c in cot:
            with st.expander(f"Chunk {c['chunk_id'] + 1}"):
                st.write(c["reasoning"])

# ------------------------------------------------------------------ compare profiles
with tabs[7]:
    st.caption("Same lecture, different learner: proof that the profile conditions the output.")
    other_name = st.selectbox("Compare with", list(profile_mod.PRESETS))
    if st.button("Generate notes for this profile", disabled=not providers):
        other = profile_mod.PRESETS[other_name]
        opts = KitOptions(**{**options.__dict__, "outputs": ("notes",)})
        try:
            ss.compare = (other_name, run_with_progress(build_kit, meta.id, other, opts))
        except Exception as exc:
            st.error(f"Generation failed: {exc}")
    if ss.compare and ss.compare[1]["meta"].id == meta.id:
        import textstat

        def stats_of(n):
            text = notes_mod.notes_to_text(n)
            return (len(text.split()), sum(len(s.bullets) for s in n.sections),
                    round(textstat.flesch_kincaid_grade(text), 1))

        left, right = st.columns(2)
        for col, (label, n) in zip((left, right), (("Current profile", kit["notes"]),
                                                   (ss.compare[0], ss.compare[1]["notes"]))):
            words, bullets, grade = stats_of(n)
            with col:
                st.markdown(f"### {label}")
                a, b, c = st.columns(3)
                a.metric("Words", words)
                b.metric("Bullets", bullets)
                c.metric("FK grade", grade)
                st.markdown(export.notes_to_markdown(n))

# ------------------------------------------------------------------ transcript
with tabs[8]:
    segments = store.load_transcript(meta.id) or []
    query = st.text_input("Search the transcript")
    rows = [{"time": fmt_ts(s.start), "text": s.text} for s in segments
            if not query or query.lower() in s.text.lower()]
    with st.form("jump_form", border=False):
        jump_col, go_col = st.columns([4, 1])
        jump = jump_col.text_input("Jump the player to (MM:SS)")
        if go_col.form_submit_button("Jump") and parse_ts(jump) is not None:
            seek_to(parse_ts(jump))
            st.rerun()
    st.dataframe(rows, use_container_width=True, hide_index=True, height=520)
    st.download_button("⬇ Transcript (JSON)", json.dumps([s.model_dump() for s in segments], ensure_ascii=False, indent=1),
                       f"{meta.title}_transcript.json", "application/json")
