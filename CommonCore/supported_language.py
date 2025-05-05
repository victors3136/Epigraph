from enum import Enum

class SupportedLanguage(Enum):
    Italian = 'ita'
    Spanish = 'spa'

    def to_best_model(self) -> str:
        return f'best_{self}_model.pt'

    def __str__(self):
        return self.value