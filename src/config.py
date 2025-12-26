from typing import Dict
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class EditorConfig:
    promts_yaml: str = "prompts.yaml"

    # Конфиг модели
    model_name: str = "gpt-5"  # замените на называние вашей модели
    base_url: str = "https://api.openai.com/v1"  # замените на ваш хост LLM если надо
    temperature: float = 0.5
    context_length: int = 16384
    llm_timeout: int = 300
    count_tokens_for: str = "gpt-5"  # Для чего считаем токены
    prices: Dict = field(
        default_factory=lambda: {
            "gpt-5": {"input": 1.25, "output": 10.00},
            "gpt-4o": {"input": 2.50, "output": 10.00},
            "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        }
    )  # расценки моделей. можете указать другие цены или модели

    # База данных разбитых на чанки уроков курса
    # где искать курс / как правило нужно указать ту же папку куда и качается
    lessons_folder: str = "course"
    # адрес хранения базы данных с чанками уроков
    course_db_path: str = "./lessons_db"
    # включаем память или без неё (не для всех методов актуально)
    enable_lessons_memory: bool = True
    # каждый раз перезаписывает базу при инициализации
    force_update_lessons: bool = False
    # сколько чанков истории подтягивает
    top_k_lessons: int = 3

    # Downloader
    # откройте курс в браузере и в адресе увидете его ID https://stepik.org/course/123456
    course_id: int = 123456
    course_save_folder: str = "course"  # куда скачать куср
    encoding: str = "utf-8"  # кодировка
    # скачивание изображений и замена ссылок на локальные изображения
    download_images: bool = False
    # добавление head для простого стиля отображения уроков в браузере
    html_head: bool = False
