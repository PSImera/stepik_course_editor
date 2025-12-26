import yaml
import re
from pathlib import Path
from datetime import datetime
from typing import List, Tuple

import tiktoken
from langchain_openai import ChatOpenAI
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from src.config import EditorConfig
from src.lesson_memory import LessonMemory

import warnings

warnings.filterwarnings("ignore")


class LessonsEditor:
    def __init__(self, config: EditorConfig = EditorConfig()):
        self.config = config
        self.llm = ChatOpenAI(
            base_url=self.config.base_url,
            model=self.config.model_name,
            temperature=self.config.temperature,
            timeout=self.config.llm_timeout,
        )

        with open(config.promts_yaml, "r", encoding="utf-8") as f:
            self.prompts = yaml.safe_load(f)

        if config.enable_lessons_memory:
            print("\n📚 Инициализация базы уроков курса...")
            self.lessons_db = LessonMemory()
            print("✅ База уроков готова")

    def _find_html_files(self, directory: str) -> List[Path]:
        """Поиск всех HTML файлов в директории"""
        html_files = []
        path = Path(directory)

        if not path.exists():
            raise ValueError(f"Папка не существует: {directory}")

        for file_path in path.rglob("*.html"):
            html_files.append(file_path)
        for file_path in path.rglob("*.htm"):
            html_files.append(file_path)

        return sorted(html_files)

    def _read_file(self, file_path: Path) -> str:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except UnicodeDecodeError:
            for encoding in ["cp1251", "latin-1", "iso-8859-1"]:
                try:
                    with open(file_path, "r", encoding=encoding) as f:
                        return f.read()
                except UnicodeDecodeError:
                    continue
            raise ValueError(f"Не удалось прочитать файл {file_path}")

    def _clean_markdown(self, text: str) -> str:
        """Удаляет markdown форматирование из ответа LLM"""
        text = text.strip()
        if text.startswith("```html"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()

    def _make_prompt(self, content, coords):
        """Собирает промпт из шаблона"""

        system_current = self.prompts["rewrite_lesson"]["system"]["current"]
        task_current = self.prompts["rewrite_lesson"]["task"]["current"]
        system_prompt = self.prompts["rewrite_lesson"]["system"][system_current]
        task_prompt = self.prompts["rewrite_lesson"]["task"][task_current]

        from_memory_context = ""
        if self.config.enable_lessons_memory:
            print(f"   📚 Получение контекста предыдущих уроков. {coords}")
            from_memory_context = self.lessons_db.prev_lessons_context(
                coords, self.config.top_k_lessons
            )
            if from_memory_context:
                rag_lessons_context_tokens = self._estimate_tokens(from_memory_context)
                print(
                    f"   ✅ Добавлен контекст уроков ({rag_lessons_context_tokens} токенов)"
                )
            else:
                print("   ℹ️ Контекст предыдущих уроков не найден")

        # Формирование итогового промпта
        chat_template = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(system_prompt),
                HumanMessagePromptTemplate.from_template(task_prompt),
            ]
        )

        prompt = chat_template.invoke(
            {"lesson": content, "lessons_memory": from_memory_context}
        )

        return prompt

    def _clean_heading(self, text: str) -> str:
        """Очистка заготловка от эмоджи символов"""
        text = re.sub(r"[🔹🔸✨🎯📌💡🚀⚡🔥]+\s*", "", text)
        return text.strip()

    def _parse_file_structure(
        self, file_path: Path
    ) -> Tuple[int, int, int, str, str, str]:
        """
        Парсит int номера модуля, урока, шага и их названия из пути файла
        """
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

    def _estimate_tokens(self, text):
        """
        Рассчитывает количество токенов в тексте
        """
        if self.config.count_tokens_for:
            try:
                enc = tiktoken.encoding_for_model(self.config.count_tokens_for)
            except:
                enc = tiktoken.get_encoding("cl100k_base")
        else:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))

    def _calculate_cost(self, input_tokens: int, output_tokens: int = 0) -> int:
        """
        Считает ценe запроса
        :param input_tokens: количество токенов в вопросе
        :param output_tokens: количество токенов в ответе
        """
        prices = self.config.prices[self.config.count_tokens_for]
        cost_input = (input_tokens / 1_000_000) * prices["input"]
        cost_output = (output_tokens / 1_000_000) * prices["output"]
        return cost_input + cost_output

    def _process_single_file(
        self, file_path: Path, structure: Tuple[int, int, int, str, str, str]
    ):
        """
        Редактирование одного файла

        :param file_path: путь к файлу
        :param structure: номера модуля, урока, шага и их названия
        """
        try:
            print(f"\n📄 {file_path.name}")
            module_num, lesson_num, step_num, _, _, _ = structure

            text_content = self.lessons_db._parse_lesson_html(file_path)
            chunks = self.lessons_db._split_lesson_text(
                text_content, file_path, structure
            )
            print(
                f"   📖 Загружен урок (Позиция: {module_num}-{lesson_num}-{step_num})"
            )

            edited_chanks = []
            edited_chanks_count = 0
            total_cost = 0
            print(f"   📖 Урок разбит на {len(chunks)} частей")
            for i, chunk in enumerate(chunks):
                print(f"\n      Обрабатываем часть [{i+1}/{len(chunks)}]\n")
                md = chunk.metadata
                coords = (
                    md["module_num"],
                    md["lesson_num"],
                    md["step_num"],
                    md["section_num"],
                    md["subsection_num"],
                )
                prompt = self._make_prompt(chunk.page_content, coords)

                estimated_tokens = self._estimate_tokens(
                    "".join([m.content for m in prompt.to_messages()])
                )
                context_info = (
                    " (с контекстом)" if self.config.enable_lessons_memory else ""
                )
                print(
                    f"      📏 Размер промпта{context_info}: {estimated_tokens} токенов"
                )

                max_input_tokens = self.config.context_length - 2500
                if estimated_tokens > max_input_tokens:
                    raise Exception(
                        f"      ⚠️ Контекст слишком большой: ({estimated_tokens} токенов, лимит - {max_input_tokens})"
                    )

                print(f"      🔄 Отправляю запрос LLM...")
                response = self.llm.invoke(prompt)
                total_cost += self._calculate_cost(
                    response.usage_metadata["input_tokens"],
                    response.usage_metadata["output_tokens"],
                )

                edited_chunk = self._clean_markdown(response.content)
                edited_chanks.append(edited_chunk)

                # Добавляем отредактированный чанк в БД уроков
                if self.config.enable_lessons_memory and self.lessons_db:
                    print(f"      💾 Сохранение чанка в RAG...")
                    chunk.page_content = edited_chunk
                    chunk.metadata.update({"timestamp": datetime.now().isoformat()})
                    self.lessons_db._add_chunk_to_db(chunk)

                # Проверяем изменения
                if chunk.page_content != edited_chunk:
                    edited_chanks_count += 1

            # Сохраняем файл
            edited_lesson = "\n".join(edited_chanks)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(edited_lesson)

            print(f"✅ Изменён")
            print(f"💵 Цена запроса\ответа. {total_cost}$")
        except Exception as e:
            print(f"❌ {str(e)}")

    def process_file(self, module_num: int, lesson_num: int, step_num: int):
        """
        Редактировать один шаг. Разбивает шаг на чанки по h1/h2 заголовкам,
        обрабатывает поочерёдно давая модели в контексте предыдущие k чанков в т.ч. с предыдущих шагов

        :param module_num: номер модуля
        :param lesson_num: номер урока
        :param step_num: номер шага
        """
        html_files = self._find_html_files(self.config.course_save_folder)
        for file_path in html_files:
            structure = self._parse_file_structure(file_path)
            if (
                structure[0] == module_num
                and structure[1] == lesson_num
                and structure[2] == step_num
            ):
                found_file = file_path
                break
        self._process_single_file(found_file, structure)

    def edit_text(self, module_num: int, lesson_num: int, step_num: int):
        """
        Просто редактирование урока, без всякого контекста и разбиения
        Больше подходит для заданий вроде проверки ошибок, нежели чем для полного переписывания с сложнымзаданием

        :param module_num: номер модуля
        :param lesson_num: номер урока
        :param step_num: номер шага
        """
        html_files = self._find_html_files(self.config.course_save_folder)
        for file_path in html_files:
            structure = self._parse_file_structure(file_path)
            if (
                structure[0] == module_num
                and structure[1] == lesson_num
                and structure[2] == step_num
            ):
                found_file = file_path
                break
        try:
            module_num, lesson_num, step_num, _, _, _ = structure

            with open(found_file, "r", encoding="utf-8") as f:
                content = f.read()

            system_current = self.prompts["edit_text"]["system"]["current"]
            task_current = self.prompts["edit_text"]["task"]["current"]
            system_prompt = self.prompts["edit_text"]["system"][system_current]
            task_prompt = self.prompts["edit_text"]["task"][task_current]

            chat_template = ChatPromptTemplate.from_messages(
                [
                    SystemMessagePromptTemplate.from_template(system_prompt),
                    HumanMessagePromptTemplate.from_template(task_prompt),
                ]
            )

            prompt = chat_template.invoke({"lesson": content})

            estimated_tokens = self._estimate_tokens(
                "".join([m.content for m in prompt.to_messages()])
            )
            print(f"      📏 Размер промпта: {estimated_tokens} токенов")

            max_input_tokens = self.config.context_length - 2500
            if estimated_tokens > max_input_tokens:
                raise Exception(
                    f"      ⚠️ Контекст слишком большой: ({estimated_tokens} токенов, лимит - {max_input_tokens})"
                )

            print(f"      🔄 Отправляю запрос LLM...")
            response = self.llm.invoke(prompt)
            total_cost = self._calculate_cost(
                response.usage_metadata["input_tokens"],
                response.usage_metadata["output_tokens"],
            )

            with open(found_file, "w", encoding="utf-8") as f:
                f.write(response.content)

            print(f"✅ Изменён")
            print(
                f"💵 Цена запроса\ответа. {total_cost}$, {self._rub(total_cost)}RUB, {self._amd(total_cost)}AMD"
            )
        except Exception as e:
            print(f"❌ {str(e)}")

    def ask_about_text(self, module_num: int, lesson_num: int, step_num: int):
        """
        Даётся в контексте текст шага и в соответствии с промптом модели отвечает.
        подходит для заданий вроде "составь тесты"

        :param module_num: номер модуля
        :param lesson_num: номер урока
        :param step_num: номер шага
        """
        html_files = self._find_html_files(self.config.course_save_folder)
        for file_path in html_files:
            structure = self._parse_file_structure(file_path)
            if (
                structure[0] == module_num
                and structure[1] == lesson_num
                and structure[2] == step_num
            ):
                found_file = file_path
                break
        try:
            module_num, lesson_num, step_num, _, _, _ = structure

            with open(found_file, "r", encoding="utf-8") as f:
                content = f.read()

            system_current = self.prompts["ask_about_text"]["system"]["current"]
            task_current = self.prompts["ask_about_text"]["task"]["current"]
            system_prompt = self.prompts["ask_about_text"]["system"][system_current]
            task_prompt = self.prompts["ask_about_text"]["task"][task_current]

            chat_template = ChatPromptTemplate.from_messages(
                [
                    SystemMessagePromptTemplate.from_template(system_prompt),
                    HumanMessagePromptTemplate.from_template(task_prompt),
                ]
            )

            prompt = chat_template.invoke({"lesson": content})

            estimated_tokens = self._estimate_tokens(
                "".join([m.content for m in prompt.to_messages()])
            )
            print(f"      📏 Размер промпта: {estimated_tokens} токенов")

            max_input_tokens = self.config.context_length - 2500
            if estimated_tokens > max_input_tokens:
                raise Exception(
                    f"      ⚠️ Контекст слишком большой: ({estimated_tokens} токенов, лимит - {max_input_tokens})"
                )

            print(f"      🔄 Отправляю запрос LLM...")
            response = self.llm.invoke(prompt)
            total_cost = self._calculate_cost(
                response.usage_metadata["input_tokens"],
                response.usage_metadata["output_tokens"],
            )
            print(f"💵 Цена запроса\ответа. {total_cost}$")
            print("\n\n", response.content, "\n\n")

        except Exception as e:
            print(f"❌ {str(e)}")


if __name__ == "__main__":
    agent = LessonsEditor()
    agent.process_file(module_num=1, lesson_num=1, step_num=1)
