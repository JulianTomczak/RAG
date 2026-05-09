import json
from src.config import BASE_DIR, DEFAULT_SUBJECT_PROMPT

SUBJECTS_FILE = BASE_DIR / "subjects.json"


def _migrate():
    try:
        with open(SUBJECTS_FILE, encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return
    if not data or isinstance(data[0], dict):
        return
    new_data = [{"name": name, "prompt": DEFAULT_SUBJECT_PROMPT} for name in data]
    with open(SUBJECTS_FILE, "w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)


_migrate()


def _load():
    try:
        with open(SUBJECTS_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _save(subjects):
    with open(SUBJECTS_FILE, "w", encoding="utf-8") as f:
        json.dump(subjects, f, ensure_ascii=False, indent=2)


def create_subject(name: str, prompt: str = None) -> bool:
    subjects = _load()
    existing_names = [s["name"] for s in subjects]
    if name and name not in existing_names:
        subjects.append({"name": name, "prompt": prompt or DEFAULT_SUBJECT_PROMPT})
        _save(subjects)
        return True
    return False


def list_subjects():
    return [s["name"] for s in _load()]


def get_subject_prompt(name: str) -> str | None:
    subjects = _load()
    for s in subjects:
        if s["name"] == name:
            return s.get("prompt")
    return None


def update_subject_prompt(name: str, prompt: str) -> bool:
    subjects = _load()
    for s in subjects:
        if s["name"] == name:
            s["prompt"] = prompt
            _save(subjects)
            return True
    return False


def delete_subject(name: str):
    subjects = _load()
    subjects = [s for s in subjects if s["name"] != name]
    _save(subjects)
    from src.repository import delete_by_materia
    from src.file_index import list_files, remove_file
    delete_by_materia(name)
    for entry in list_files():
        if entry["materia"] == name:
            remove_file(entry["filename"])
