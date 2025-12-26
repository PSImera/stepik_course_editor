from pathlib import Path
import re
import json
import pickle
from datetime import datetime
from typing import Tuple
from collections import defaultdict

from langchain_core.documents import Document
from bs4 import BeautifulSoup

from src.config import EditorConfig


def make_tree():
    return defaultdict(make_tree)


class LessonMemory:
    def __init__(self, config: EditorConfig = EditorConfig()):
        self.config = config

        self.log_messages = []
        self._setup_storage_paths()
        self._init_course_db()

    def _setup_storage_paths(self):
        self.lessons_folder = Path(self.config.lessons_folder)

        self.lessons_db_path = Path(self.config.course_db_path)
        self.lessons_db_path.mkdir(parents=True, exist_ok=True)

        self.log_path = self.lessons_db_path / "lessons.log"
        self.db_metadata_path = self.lessons_db_path / "db_metadata.json"
        self.course_db_path = self.lessons_db_path / "course_db.pkl"

    def _init_course_db(self):
        if self._should_update_db():
            self._build_course_db()
        else:

            self._load_course_db()

    def _should_update_db(self) -> bool:
        if self.config.force_update_lessons:
            self._log("=" * 60)
            self._log("📚 ПРИНУДИТЕЛЬНОЕ ОБНОВЛЕНИЕ БАЗЫ КУРСА")
            self._log("=" * 60)
            return True

        self._load_db_metadata()

        if not self.db_metadata.get("last_update"):
            self._log("=" * 60)
            self._log("📚 СОЗДАНИЕ БАЗЫ КУРСА")
            self._log("=" * 60)
            return True

        if not self.course_db_path.exists():
            self._log("=" * 60)
            self._log("📚 СОЗДАНИЕ БАЗЫ КУРСА")
            self._log("=" * 60)
            return True

        last_update = datetime.fromisoformat(self.db_metadata["last_update"])
        days_old = (datetime.now() - last_update).days
        print(f"✅ Используем существуюую базу курса")
        print(f"💬 БД обновлена {days_old} дней назад")
        print(f"💬 Уроков в БД: {self.db_metadata.get('lessons_count', 0)}")
        print(f"💬 Чанков в БД: {self.db_metadata.get('chunks_count', 0)}")

        return False

    def _load_course_db(self):
        try:
            with open(self.course_db_path, "rb") as f:
                self.course_db = pickle.load(f)
            self._load_db_metadata()

        except Exception as e:
            self._log(f"⚠️ Ошибка загрузки Чанков уроков: {e}", "ERROR")
            self.course_db = make_tree()

    def _save_course_db(self):
        try:
            with open(self.course_db_path, "wb") as f:
                pickle.dump(self.course_db, f)
            self._save_db_metadata()
        except Exception as e:
            self._log(f"   ⚠️ Ошибка сохранения базы уроков: {e}", "ERROR")

    def _add_chunk_to_db(self, chunk):
        module = chunk.metadata.get("module_num")
        lesson = chunk.metadata.get("lesson_num")
        step = chunk.metadata.get("step_num")
        section = chunk.metadata.get("section_num")
        subsection = chunk.metadata.get("subsection_num")
        self.course_db[module][lesson][step][section][subsection] = chunk
        self._save_course_db()

    def _add_file_to_db(self, file_path):
        try:
            structure = self._parse_file_structure(file_path)
            text_content = self._parse_lesson_html(file_path)
            chunks = self._split_lesson_text(text_content, file_path, structure)

            if not chunks:
                self._log(f"   ⚠️ В уроке не найдено Чанков", "ERROR")
                return
            for chunk in chunks:
                self._add_chunk_to_db(chunk)

        except Exception as e:
            self._log(f"   ⚠️ Ошибка обработки урока: {e}", "ERROR")
            return

    def _build_course_db(self):
        if not self.lessons_folder.exists():
            self._log(f"❌ Папка с уроками не найдена: {self.lessons_folder}", "ERROR")
            return

        self._log(f"📂 Считываем уроки в базу данных")
        self.course_db = make_tree()
        html_files = sorted(self.lessons_folder.rglob("*.html"))
        self._log(f"📂 Найдено HTML файлов: {len(html_files)}")

        for idx, file_path in enumerate(html_files, 1):
            self._log(f"📄 [{idx}/{len(html_files)}] {file_path.name}")
            self._add_file_to_db(file_path)

        self._save_course_db()
        self.print_statistics()

    def _iter_db(self, db):
        if isinstance(db, Document):
            yield db
        elif isinstance(db, dict):
            for value in db.values():
                yield from self._iter_db(value)

    def _load_db_metadata(self):
        if self.db_metadata_path.exists():
            with open(self.db_metadata_path, "r", encoding="utf-8") as f:
                self.db_metadata = json.load(f)
        else:
            self.db_metadata = {}

    def _save_db_metadata(self):
        try:
            self.db_metadata = {
                "last_update": datetime.now().isoformat(),
                "lessons_count": sum(
                    len(self.course_db[module]) for module in self.course_db
                ),
                "chunks_count": sum(1 for _ in self._iter_db(self.course_db)),
                "chunks_metadata": [d.metadata for d in self._iter_db(self.course_db)],
            }
            with open(self.db_metadata_path, "w", encoding="utf-8") as f:
                json.dump(self.db_metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self._log(f"⚠️ Ошибка сохранения метаданных: {e}", "ERROR")

    def _log(self, message: str, level: str = "INFO"):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}"
        self.log_messages.append(log_entry)
        print(message)
        self._save_log()

    def _save_log(self):
        if not self.log_messages:
            return
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write("\n".join(self.log_messages) + "\n")
        self.log_messages = []

    def _clean_heading(self, text: str) -> str:
        text = re.sub(r"[🔹🔸✨🎯📌💡🚀⚡🔥]+\s*", "", text)
        return text.strip()

    def _parse_file_structure(
        self, file_path: Path
    ) -> Tuple[int, int, int, str, str, str]:
        filename = file_path.stem

        match = re.match(r"(\d+)-(\d+)-(\d+)_(.+)", filename)
        if not match:
            raise ValueError(f"Неверный формат имени файла: {filename}")

        module_num = int(match.group(1))
        lesson_num = int(match.group(2))
        step_num = int(match.group(3))
        step_title = match.group(4).strip()
        step_title = self._clean_heading(step_title)

        lesson_folder = file_path.parent
        lesson_match = re.match(r"(\d+)\.\s*(.+)", lesson_folder.name)
        if lesson_match:
            folder_lesson_num = int(lesson_match.group(1))
            lesson_title = lesson_match.group(2).strip()

            if folder_lesson_num != lesson_num:
                print(
                    f"⚠️ Несоответствие номера урока: файл={lesson_num}, папка={folder_lesson_num} ({lesson_folder.name})"
                )
        else:
            lesson_title = lesson_folder.name
            print(f"⚠️ Неверный формат папки урока: {lesson_folder.name}")

        module_folder = lesson_folder.parent
        module_match = re.match(r"(\d+)\.\s*(.+)", module_folder.name)
        if module_match:
            folder_module_num = int(module_match.group(1))
            module_title = module_match.group(2).strip()

            if folder_module_num != module_num:
                print(
                    f"⚠️ Несоответствие номера модуля: файл={module_num}, папка={folder_module_num} ({module_folder.name})"
                )
        else:
            module_title = module_folder.name
            print(f"⚠️ Неверный формат папки модуля: {module_folder.name}")

        return module_num, lesson_num, step_num, module_title, lesson_title, step_title

    def _parse_lesson_html(self, file_path: str) -> str:
        with open(file_path, "r", encoding="utf-8") as f:
            html_content = f.read()

        soup = BeautifulSoup(html_content, "html.parser")
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()

        content_div = soup.find("main") or soup.find("body") or soup

        for h in content_div.find_all(["h2", "h3"]):
            if not h.get_text(strip=True):
                h.decompose()

        for h2 in content_div.find_all("h2"):
            h2.string = f"\n\n##h2## {self._clean_heading(h2.get_text(strip=True))}\n\n"

        for h3 in content_div.find_all("h3"):
            h3.string = (
                f"\n\n###h3### {self._clean_heading(h3.get_text(strip=True))}\n\n"
            )

        for pre in content_div.find_all("pre"):
            code = pre.find("code")
            if code:
                pre.string = f"\n```\n{code.get_text()}\n```\n"
            else:
                pre.string = f"\n```\n{pre.get_text()}\n```\n"

        for br in content_div.find_all("br"):
            br.replace_with("\n")

        for ul in content_div.find_all("ul"):
            for li in ul.find_all("li", recursive=False):
                p = li.find("p")
                if p:
                    li.string = f"- {p.get_text(strip=True)}"
                else:
                    li.string = f"- {li.get_text(strip=True)}"

        for ol in content_div.find_all("ol"):
            counter = 1
            for li in ol.find_all("li", recursive=False):
                p = li.find("p")
                if p:
                    li.string = f"{counter}. {p.get_text(strip=True)}"
                else:
                    li.string = f"{counter}. {li.get_text(strip=True)}"
                counter += 1

        text_content = content_div.get_text()
        text_content = re.sub(r"\n\s*\n\s*\n+", "\n\n", text_content)

        return text_content.strip()

    def _split_lesson_text(
        self,
        text_content: str,
        file_path: str,
        structure: Tuple[int, int, int, str, str, str],
    ):
        module_num, lesson_num, step_num, module_title, lesson_title, step_title = (
            structure
        )

        chunks = []
        current_section_num = 1
        current_subsection_num = 1
        current_section_title = None
        current_subsection_title = None
        buffer_lines = []
        if not step_title:
            step_title = f"Шаг {step_num}"

        h2_count = 0

        def make_metadata():
            return {
                "module_num": module_num,
                "lesson_num": lesson_num,
                "step_num": step_num,
                "section_num": current_section_num,
                "subsection_num": current_subsection_num,
                "module_title": module_title,
                "lesson_title": lesson_title,
                "step_title": step_title,
                "section_title": current_section_title,
                "subsection_title": current_subsection_title,
                "file_path": str(file_path),
                "timestamp": datetime.now().isoformat(),
                "doc_type": "lesson",
            }

        for line in text_content.split("\n"):
            if not line:
                continue

            if line.startswith("##h2## "):
                h2_count += 1

                if buffer_lines:
                    chunks.append(
                        Document(
                            page_content="\n".join(buffer_lines).strip(),
                            metadata=make_metadata(),
                        )
                    )
                    buffer_lines = []

                section_title = line.replace("##h2## ", "")

                if h2_count == 1:
                    current_section_num = 1
                    current_section_title = None
                else:
                    current_section_num += 1
                    current_section_title = section_title

                current_subsection_num = 1
                current_subsection_title = None
                continue

            if line.startswith("###h3### "):
                if buffer_lines:
                    chunks.append(
                        Document(
                            page_content="\n".join(buffer_lines).strip(),
                            metadata=make_metadata(),
                        )
                    )
                    buffer_lines = []

                current_subsection_num += 1
                current_subsection_title = line.replace("###h3### ", "")
                continue

            buffer_lines.append(line)

        if buffer_lines:
            chunks.append(
                Document(
                    page_content="\n".join(buffer_lines).strip(),
                    metadata=make_metadata(),
                )
            )

        return chunks

    def _get_prev_chunk(self, db, coords):
        module, lesson, step, section, subsection = coords

        if subsection > 1:
            if (
                module in db
                and lesson in db[module]
                and step in db[module][lesson]
                and section in db[module][lesson][step]
            ):
                if subsection - 1 in db[module][lesson][step][section]:
                    return db[module][lesson][step][section][subsection - 1]
            return self._get_prev_chunk(
                db, (module, lesson, step, section, subsection - 1)
            )

        if section > 1:
            if module in db and lesson in db[module] and step in db[module][lesson]:
                sections = db[module][lesson][step]
                prev_section_keys = sorted([k for k in sections.keys() if k < section])
                if prev_section_keys:
                    prev_section = prev_section_keys[-1]
                    prev_subsections = sections[prev_section]
                    last_sub = max(prev_subsections.keys())
                    return prev_subsections[last_sub]
            return self._get_prev_chunk(
                db, (module, lesson, step - 1, float("inf"), float("inf"))
            )

        if step > 1:
            if module in db and lesson in db[module]:
                steps = db[module][lesson]
                prev_step_keys = sorted([k for k in steps.keys() if k < step])
                if prev_step_keys:
                    prev_step = prev_step_keys[-1]
                    prev_sections = steps[prev_step]
                    last_section = max(prev_sections.keys())
                    last_sub = max(prev_sections[last_section].keys())
                    return prev_sections[last_section][last_sub]
            return self._get_prev_chunk(
                db, (module, lesson - 1, float("inf"), float("inf"), float("inf"))
            )

        if lesson > 1:
            if module in db:
                lessons = db[module]
                prev_lesson_keys = sorted([k for k in lessons.keys() if k < lesson])
                if prev_lesson_keys:
                    prev_lesson = prev_lesson_keys[-1]
                    prev_steps = lessons[prev_lesson]
                    last_step = max(prev_steps.keys())
                    prev_sections = prev_steps[last_step]
                    last_section = max(prev_sections.keys())
                    last_sub = max(prev_sections[last_section].keys())
                    return prev_sections[last_section][last_sub]
            return self._get_prev_chunk(
                db, (module - 1, float("inf"), float("inf"), float("inf"), float("inf"))
            )

        if module > 1:
            prev_module_keys = sorted([k for k in db.keys() if k < module])
            if prev_module_keys:
                prev_module = prev_module_keys[-1]
                prev_lessons = db[prev_module]
                last_lesson = max(prev_lessons.keys())
                prev_steps = prev_lessons[last_lesson]
                last_step = max(prev_steps.keys())
                prev_sections = prev_steps[last_step]
                last_section = max(prev_sections.keys())
                last_sub = max(prev_sections[last_section].keys())
                return prev_sections[last_section][last_sub]

        return None

    def print_statistics(self):
        self._log("=" * 60)
        self._log(f"📊 СТАТИСТИКА БАЗЫ УРОКОВ")
        self._log("=" * 60)
        self._log(f"📊   Уроков: {self.db_metadata['lessons_count']}")
        self._log(f"📊   Чанков: {self.db_metadata['chunks_count']}")
        last_m = self.db_metadata["chunks_metadata"][-1]["module_num"]
        last_l = self.db_metadata["chunks_metadata"][-1]["lesson_num"]
        last_s = self.db_metadata["chunks_metadata"][-1]["step_num"]
        self._log(f"📊   Последний урок: ({last_m}-{last_l}-{last_s})")
        self._log("=" * 60)
        self._log("")

    def prev_lessons_context(self, coords, top_k=None):
        if not top_k:
            top_k = self.config.top_k_lessons
        collected = []
        cur_coords = coords

        for _ in range(top_k):
            chunk = self._get_prev_chunk(self.course_db, cur_coords)
            if not chunk:
                break
            collected.append(chunk)

            md = chunk.metadata
            cur_coords = (
                md["module_num"],
                md["lesson_num"],
                md["step_num"],
                md["section_num"],
                md["subsection_num"],
            )

        if not collected:
            return ""

        collected.reverse()
        context_parts = []

        prev_module = coords[0]
        prev_lesson = coords[1]
        prev_step = coords[2]
        prev_section = coords[3]
        prev_subsection = coords[4]

        for doc in collected:
            module_title = doc.metadata["module_title"]
            lesson_title = doc.metadata["lesson_title"]
            step_title = doc.metadata["step_title"]
            section_title = doc.metadata["section_title"]
            subsection_title = doc.metadata["subsection_title"]

            if prev_step != doc.metadata["step_num"]:
                title_text = f"⏮️  На прошлом шаге ({step_title})"
                if prev_lesson != doc.metadata["lesson_num"]:
                    title_text += f", прошлго урока ({lesson_title})"
                    if prev_module != doc.metadata["module_num"]:
                        title_text += f", модуля ({module_title})\n"
                    else:
                        title_text += "\n"
                else:
                    title_text += "\n"
            else:
                title_text = ""

            if prev_section != doc.metadata["section_num"] and section_title:
                title_text += f"🔹 {section_title}\n"

            if prev_subsection != doc.metadata["subsection_num"] and subsection_title:
                title_text += f"🔸 {subsection_title}\n"

            context_parts.append(f"{title_text}{doc.page_content}")

            prev_module = doc.metadata["module_num"]
            prev_lesson = doc.metadata["lesson_num"]
            prev_step = doc.metadata["step_num"]
            prev_section = doc.metadata["section_num"]
            prev_subsection = doc.metadata["subsection_num"]

        context = "=== ДО ЭТОГО В КУРСЕ МЫ УЖЕ ОБСУЖДАЛИ СЛЕДУЮЩЕЕ:\n\n"
        context += "\n\n".join(context_parts)
        context += "\n\n=== КОНЕЦ ЦИТИРОВАНИЯ ИНФОРМАЦИИ ИЗ КУРСА"

        return context


def main():
    lessons_db = LessonMemory()

    text = lessons_db.prev_lessons_context((5, 1, 1, 1, 1), 5)
    print(text)


if __name__ == "__main__":
    main()
