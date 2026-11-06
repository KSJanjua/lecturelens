"""Cloze and Q/A flashcards from the notes, exportable as an Anki-importable CSV."""
import csv
import io

import config
from pipeline import llm, prompts
from pipeline.notes import notes_to_text
from pipeline.schemas import Flashcard, FlashcardSet, LearnerProfile, Notes


def generate_flashcards(notes: Notes, profile: LearnerProfile, max_cards: int | None = None) -> FlashcardSet:
    max_cards = max_cards or config.FLASHCARD_MAX
    glossary = "\n".join(f"- {g.term}: {g.definition}" for g in notes.glossary)
    text = notes_to_text(notes, with_summary=False) + (f"\n\nGlossary:\n{glossary}" if glossary else "")
    result = llm.generate_json(
        prompts.render("flashcards", max_cards=max_cards, profile=profile.describe(), notes=text,
                       language=profile.language),
        FlashcardSet, system=prompts.system(), temperature=0.3, task="flashcards",
    )
    seen, cards = set(), []
    for card in result.cards:
        key = card.front.strip().lower()
        if key and key not in seen and card.back.strip():
            seen.add(key)
            if card.kind == "cloze" and "_____" not in card.front and card.back in card.front:
                card.front = card.front.replace(card.back, "_____", 1)
            cards.append(card)
    return FlashcardSet(cards=cards[:max_cards])


def to_anki_csv(cards: list[Flashcard], deck_tag: str = "LectureLens") -> str:
    """Front, Back, Tags. Import in Anki with the 'Basic' note type, fields separated by comma."""
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    tag = deck_tag.replace(" ", "_")
    for card in cards:
        back = card.back + (f" [{card.ts}]" if card.ts else "")
        writer.writerow([card.front, back, f"{tag} {card.kind}"])
    return buffer.getvalue()
