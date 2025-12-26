import os
from dotenv import load_dotenv
import requests
import re
from urllib.parse import urlparse
import warnings

warnings.filterwarnings("ignore")

from src.config import EditorConfig

load_dotenv()


class CourseDownloader:
    def __init__(self, config: EditorConfig = EditorConfig()):
        self.config = config

        client_id = os.getenv("CLIENT_ID")
        client_secret = os.getenv("CLIENT_SECRET")
        if not all([client_id, client_secret]):
            raise ValueError("CLIENT_ID и CLIENT_SECRET должны быть указаны в .env")
        token = self._get_token(client_id, client_secret)
        self.headers = {"Authorization": f"Bearer {token}"}
        self._get_course_info()

    def _get_token(self, client_id: str, client_secret: str) -> str:
        """Получает access_token по Client ID и Client Secret"""
        url = "https://stepik.org/oauth2/token/"
        data = {
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        }
        r = requests.post(url, data=data)
        r.raise_for_status()
        token = r.json().get("access_token")
        if not token:
            raise RuntimeError(
                "Не удалось получить токен. Проверь client_id и client_secret."
            )
        return token

    def download_course(
        self, module_num: int = None, lesson_num: int = None, step_num: int = None
    ):
        """
        Скачивает курс с выбранной глубиной:
        - Без аргументов: весь курс
        - module_num: один модуль
        - module_num + lesson_num: один урок
        - module_num + lesson_num + step_num: один шаг
        """
        # Определяем диапазон модулей для обработки
        if module_num is not None:
            sections_to_process = [(module_num, self.structure[module_num - 1])]
        else:
            sections_to_process = list(enumerate(self.structure, 1))

        lesson_counter = 0

        for section_num, section in sections_to_process:
            section_title = section["section_title"]
            lessons_list = section["lessons"]

            # Определяем диапазон уроков для обработки
            if lesson_num is not None:
                lessons_to_process = [(lesson_num, lessons_list[lesson_num - 1])]
            else:
                lessons_to_process = list(enumerate(lessons_list, 1))

            # Выводим информацию о разделе
            if module_num is None:
                print(f"\n📚 Раздел {section_num}: {section_title}")
            else:
                print(f"\n📚 Раздел {section_num}: {section_title}")
                if lesson_num is None:
                    print(f"     Найдено уроков: {len(lessons_list)}\n")

            # Создаем папку для раздела
            section_folder = self._create_section_folder(section_num, section_title)

            for current_lesson_num, lesson_info in lessons_to_process:
                lesson_counter += 1
                lesson_id = lesson_info["lesson_id"]
                lesson_title = lesson_info["lesson_title"]

                # Выводим прогресс
                if lesson_num is None and module_num is None:
                    total = self.total_lessons
                elif lesson_num is None:
                    total = len(lessons_list)
                else:
                    total = 1

                if step_num is None:
                    print(
                        f"  [{lesson_counter}/{total}] Урок {current_lesson_num}: {lesson_title}"
                    )
                else:
                    print(f"    Урок {current_lesson_num}: {lesson_title}")

                # Создаем папку для урока
                lesson_folder = self._create_lesson_folder(
                    section_folder, current_lesson_num, lesson_title
                )

                # Обрабатываем шаги
                blocks = self.get_theory(lesson_id)
                if blocks:
                    # Определяем какие шаги обрабатывать
                    if step_num is not None:
                        blocks_to_process = [blocks[step_num - 1]]
                        step_numbers = [step_num]
                    else:
                        blocks_to_process = blocks
                        step_numbers = [block["position"] for block in blocks]

                    for block, current_step_num in zip(blocks_to_process, step_numbers):
                        self._process_step(
                            block,
                            lesson_folder,
                            section_num,
                            current_lesson_num,
                            current_step_num,
                        )
                else:
                    print(f"    ⚠️ Теоретических материалов не найдено")

        # Итоговое сообщение
        self._print_completion_message(
            module_num,
            lesson_num,
            step_num,
            section_folder if module_num else None,
            lesson_folder if lesson_num else None,
        )

    def _create_section_folder(self, section_num: int, section_title: str) -> str:
        """Создает и возвращает путь к папке раздела"""
        section_folder_name = (
            f"{section_num:02d}. {self.sanitize_filename(section_title)}"
        )
        section_folder = os.path.join(self.root_folder, section_folder_name)
        if not os.path.exists(section_folder):
            os.makedirs(section_folder)
        return section_folder

    def _create_lesson_folder(
        self, section_folder: str, lesson_num: int, lesson_title: str
    ) -> str:
        """Создает и возвращает путь к папке урока"""
        lesson_folder_name = f"{lesson_num:02d}. {self.sanitize_filename(lesson_title)}"
        lesson_folder = os.path.join(section_folder, lesson_folder_name)
        if not os.path.exists(lesson_folder):
            os.makedirs(lesson_folder)
        return lesson_folder

    def _process_step(
        self,
        block: dict,
        lesson_folder: str,
        section_num: int,
        lesson_num: int,
        step_num: int,
    ):
        """Обрабатывает один шаг: скачивает изображения и сохраняет HTML"""
        step_html = block["html"]

        # Скачиваем изображения для шага
        if self.config.download_images:
            step_html_with_images = self.download_images(step_html, lesson_folder)
        else:
            step_html_with_images = step_html

        # Извлекаем название шага
        step_name = self.extract_step_name(step_html)
        step_name_clean = self.sanitize_filename(step_name)

        # Формируем имя файла
        step_filename = (
            f"{section_num:02d}-{lesson_num:02d}-{step_num:02d}_{step_name_clean}.html"
        )
        step_filepath = os.path.join(lesson_folder, step_filename)

        self.save_step_html(step_html_with_images, step_filepath)
        print(f"    ✓ Шаг {step_num}: {step_name}")

    def _print_completion_message(
        self, module_num, lesson_num, step_num, section_folder, lesson_folder
    ):
        """Выводит итоговое сообщение в зависимости от глубины скачивания"""
        if step_num is not None:
            print(f"\n✅ Шаг скачан в папку: {lesson_folder}")
        elif lesson_num is not None:
            print(f"\n✅ Урок скачан в папку: {lesson_folder}")
        elif module_num is not None:
            print(f"\n✅ Модуль скачан в папку: {section_folder}")
        else:
            print(f"\n✅ Курс сохранен в папку: {self.root_folder}")

    def _get_course_info(self) -> dict:
        self.course_id = self.config.course_id
        if not self.config.course_id:
            print("❌ Не указан course_id в config.yaml")
            return

        if not all(
            hasattr(self, a) for a in ("course_info", "structure", "course_name")
        ):
            print(f"Загрузка структуры курса {self.course_id}...")

            url = f"https://stepik.org/api/courses/{self.course_id}"
            r = requests.get(url, headers=self.headers)
            r.raise_for_status()
            courses = r.json().get("courses", [])
            if courses:
                self.course_info = courses[0]
            else:
                self.course_info = {}

            self.course_name = self.sanitize_filename(
                self.course_info.get("title", f"course_{self.course_id}")
            )

            self.structure = self.get_course_structure()

            if not self.structure:
                print("❌ Не удалось получить структуру курса.")
                return
        else:
            print("Структура курса:")

        self.total_lessons = sum(len(section["lessons"]) for section in self.structure)
        print(f"Название курса: {self.course_name}")
        print(f"Разделов: {len(self.structure)}, Уроков: {self.total_lessons}")

        self.root_folder = os.path.join(
            self.config.course_save_folder, self.course_name
        )
        if not os.path.exists(self.root_folder):
            os.makedirs(self.root_folder)

    def sanitize_filename(self, name: str) -> str:
        name = re.sub(r'[<>:"/\\|?*]', "_", name)
        name = name.strip()
        if len(name) > 200:
            name = name[:200]
        return name

    def get_course_structure(self) -> list:

        structure = []
        section_ids = self.get_sections_from_course()

        if not section_ids:
            print("⚠️ Пробуем альтернативный способ получения секций...")
            sections_url = f"https://stepik.org/api/sections?course={self.course_id}"
            r = requests.get(sections_url, headers=self.headers)
            r.raise_for_status()
            sections = r.json().get("sections", [])
            section_ids = [s["id"] for s in sections]

        if not section_ids:
            print("❌ Не удалось получить секции курса")
            return []

        for section_id in section_ids:
            section_info = self.get_section_info(section_id)
            section_title = section_info.get("title", f"Раздел {section_id}")
            section_position = section_info.get("position", 0)

            unit_ids = section_info.get("units", [])
            if not unit_ids:
                print(f"  ⚠️ Секция '{section_title}' не содержит уроков")
                continue

            lessons = []
            for unit_id in unit_ids:
                unit_info = self.get_unit_info(unit_id)
                lesson_id = unit_info.get("lesson")

                if lesson_id:
                    lesson_title = self.get_lesson_title(lesson_id)
                    unit_position = unit_info.get("position", 0)
                    lessons.append(
                        {
                            "lesson_id": lesson_id,
                            "lesson_title": lesson_title,
                            "position": unit_position,
                        }
                    )

            lessons.sort(key=lambda x: x["position"])

            if lessons:
                structure.append(
                    {
                        "section_title": section_title,
                        "section_position": section_position,
                        "lessons": lessons,
                    }
                )

        structure.sort(key=lambda x: x["section_position"])

        return structure

    def get_sections_from_course(self) -> list:
        section_ids = self.course_info.get("sections", [])
        if section_ids:
            return section_ids
        else:
            print("⚠️ Список секций в курсе пуст")
            return []

    def get_section_info(self, section_id: int) -> dict:
        url = f"https://stepik.org/api/sections/{section_id}"
        r = requests.get(url, headers=self.headers)
        r.raise_for_status()
        sections = r.json().get("sections", [])
        if sections:
            return sections[0]
        return {}

    def get_unit_info(self, unit_id: int) -> dict:
        url = f"https://stepik.org/api/units/{unit_id}"
        r = requests.get(url, headers=self.headers)
        r.raise_for_status()
        units = r.json().get("units", [])
        if units:
            return units[0]
        return {}

    def get_lesson_title(self, lesson_id: int) -> str:
        try:
            url = f"https://stepik.org/api/lessons/{lesson_id}"
            r = requests.get(url, headers=self.headers)
            r.raise_for_status()
            lessons = r.json().get("lessons", [])
            if lessons:
                return lessons[0].get("title", f"Урок {lesson_id}")
        except:
            pass
        return f"Урок {lesson_id}"

    def get_theory(self, lesson_id: int) -> list[dict]:
        theory_blocks = []
        url = f"https://stepik.org/api/steps?lesson={lesson_id}"

        while url:
            r = requests.get(url, headers=self.headers)
            r.raise_for_status()
            data = r.json()

            for step in data.get("steps", []):
                block = step.get("block", {})
                if block.get("name") in ("text", "free-answer"):
                    html = block.get("text", "").strip()
                    if html:
                        step_position = step.get("position", 0)
                        theory_blocks.append({"position": step_position, "html": html})
            url = data.get("meta", {}).get("next", None)

        theory_blocks.sort(key=lambda x: x["position"])
        return theory_blocks

    def download_images(
        self, html: str, lesson_folder: str, images_folder: str = "images"
    ) -> str:
        images_path = os.path.join(lesson_folder, images_folder)
        if not os.path.exists(images_path):
            os.makedirs(images_path)

        img_pattern = re.compile(
            r'<img[^>]+src=["\']([^"\']+)["\'][^>]*>', re.IGNORECASE
        )

        def replace_img(match):
            img_tag = match.group(0)
            img_url = match.group(1)

            try:
                response = requests.get(img_url, timeout=10, verify=False)
                response.raise_for_status()

                parsed = urlparse(img_url)
                filename = os.path.basename(parsed.path)
                if not filename or "/" in filename:
                    filename = f"image_{abs(hash(img_url))}.jpg"

                filepath = os.path.join(images_path, filename)

                with open(filepath, "wb") as f:
                    f.write(response.content)

                relative_path = os.path.join(images_folder, filename).replace("\\", "/")
                new_img_tag = img_tag.replace(img_url, relative_path)
                return new_img_tag

            except Exception as e:
                print(f"⚠️ Не удалось скачать изображение {img_url}: {e}")
                return f"<!-- Изображение недоступно: {img_url} -->"

        return img_pattern.sub(replace_img, html)

    def extract_step_name(self, html: str, max_length: int = 100) -> str:
        """Извлекает название шага из HTML"""
        title = None

        # 1. Ищем заголовки h1-h6
        header_patterns = [
            r"<h1[^>]*>(.*?)</h1>",
            r"<h2[^>]*>(.*?)</h2>",
            r"<h3[^>]*>(.*?)</h3>",
            r"<h4[^>]*>(.*?)</h4>",
            r"<h5[^>]*>(.*?)</h5>",
            r"<h6[^>]*>(.*?)</h6>",
        ]

        for pattern in header_patterns:
            match = re.search(pattern, html, re.IGNORECASE | re.DOTALL)
            if match:
                title = match.group(1)
                break

        # 2. Если заголовка нет, ищем <strong> или <b> в начале
        if not title:
            strong_patterns = [r"<strong[^>]*>(.*?)</strong>", r"<b[^>]*>(.*?)</b>"]
            for pattern in strong_patterns:
                match = re.search(pattern, html, re.IGNORECASE | re.DOTALL)
                if match:
                    title = match.group(1)
                    break

        # 3. Если ничего не найдено, берем первый текстовый фрагмент
        if not title:
            # Убираем все теги
            text = re.sub(r"<[^>]+>", "", html)
            # Убираем лишние пробелы и переносы
            text = " ".join(text.split())
            if text:
                # Берем до первой точки или до max_length символов
                sentences = text.split(".")
                if sentences:
                    title = sentences[0].strip()
            else:
                title = "Untitled"

        # Убираем HTML теги из заголовка (на случай вложенных тегов)
        title = re.sub(r"<[^>]+>", "", title)
        # Убираем лишние пробелы
        title = " ".join(title.split())

        # Убираем префиксы типа "Шаг N:", "Step N:", и т.д.
        title = re.sub(
            r"^(Шаг|Step|Lesson|Урок)\s*\d+\s*[:\-\.\)]\s*",
            "",
            title,
            flags=re.IGNORECASE,
        )

        # Ограничиваем длину
        if len(title) > max_length:
            title = title[:max_length].rsplit(" ", 1)[
                0
            ]  # Обрезаем по последнему пробелу

        # Если после всех операций title пустой
        if not title or title.isspace():
            title = "Untitled"

        return title.strip()

    def save_step_html(self, step_html: str, filepath: str):
        """Сохраняет HTML шага в отдельный файл"""
        if self.config.html_head:
            full_html = f"""<html>
<head>
    <meta charset='{self.config.encoding}'>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
        img {{ max-width: 100%; height: auto; }}
    </style>
</head>
<body>
{step_html}
</body>
</html>"""
        else:
            full_html = step_html

        with open(filepath, "w", encoding=self.config.encoding) as f:
            f.write(full_html)


if __name__ == "__main__":
    config = EditorConfig()
    downloader = CourseDownloader(config)
    downloader.download_course()
