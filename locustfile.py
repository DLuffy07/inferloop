import random
import uuid

from locust import HttpUser, between, task


ARTICLES = [
    "Le PSG remporte son match de championnat après une victoire.",
    "La Banque centrale annonce une nouvelle décision sur les taux.",
    "Le gouvernement présente une nouvelle réforme devant le Parlement.",
    "Un nouveau film français sort cette semaine dans les cinémas.",
    "Un incendie s'est déclaré cette nuit dans un immeuble du centre-ville.",
]


class InferLoopUser(HttpUser):
    wait_time = between(1, 2)

    @task
    def infer_article(self):
        texte = f"{random.choice(ARTICLES)} Identifiant {uuid.uuid4()}"

        self.client.post(
            "/infer",
            json={"texte": texte},
        )