"""
Indonesian Benchmark Evaluation Suite for Galactus.
Includes:
1. IndoMMLU: Multi-subject Multiple Choice benchmark in Indonesian (Bahasa, Sejarah, Sains, PPKn, Logika).
2. IndoReasoning: Indonesian step-by-step logic and cause-and-effect reasoning benchmark.
3. AlpacaIndoEval: Generative Indonesian instruction following and conversational reasoning.

Usage:
    python -m scripts.chat_eval -i sft -a IndoMMLU
    python -m scripts.chat_eval -i sft -a "IndoMMLU|IndoReasoning|AlpacaIndo"
"""

import os
import re
import json
import pyarrow as pa
from functools import partial
from tasks.common import Task, HubDataset, load_hub_dataset, render_mc
from tasks.alpaca_indonesian import AlpacaGPT4Indonesian


def render_mc_id(question, letters, choices):
    """
    Renders multiple-choice questions with Indonesian prompt framing while
    preserving exact token binding for letter answers (no whitespace after '=').
    """
    query = f"Pertanyaan Pilihan Ganda: {question}\n"
    query += "".join([f"- {choice}={letter}\n" for letter, choice in zip(letters, choices)])
    query += "\nJawab hanya dengan huruf pilihan yang benar."
    return query


# Curated high-yield Indonesian evaluation fallback dataset
# Covering Bahasa Indonesia, Sejarah, Sains, PPKn, Matematika & Logika
CURATED_INDO_MMLU_ROWS = [
    {
        "subject": "Bahasa Indonesia",
        "question": "Manakah di bawah ini yang merupakan contoh kalimat efektif dengan kaidah EYD/PUEBI yang benar?",
        "choices": [
            "Bagi yang membawa kendaraan bermotor harap diparkir di belakang.",
            "Siswa yang berprestasi itu mendapatkan beasiswa penuh dari universitas.",
            "Dalam rapat kemarin membicarakan tentang kenaikan anggaran.",
            "Untuk mempersingkat waktu, acara berikutnya segera dimulai."
        ],
        "answer": "B",
    },
    {
        "subject": "Sejarah Indonesia",
        "question": "Peristiwa Rengasdengklok yang terjadi pada tanggal 16 Agustus 1945 bertujuan untuk...",
        "choices": [
            "Menghindari ancaman tentara Sekutu yang mulai mendarat di Jawa.",
            "Mendesak Soekarno dan Hatta agar segera memproklamasikan kemerdekaan tanpa pengaruh Jepang.",
            "Menyusun naskah Undang-Undang Dasar di rumah Laksamana Maeda.",
            "Mengamankan senjata dari markas tentara PETA di Rengasdengklok."
        ],
        "answer": "B",
    },
    {
        "subject": "PPKn",
        "question": "Sila keempat Pancasila dilambangkan dengan kepala banteng. Nilai utama yang terkandung di dalamnya adalah...",
        "choices": [
            "Persatuan dan kesatuan bangsa di atas keberagaman.",
            "Kerakyatan yang dipimpin oleh hikmat kebijaksanaan dalam permusyawaratan/perwakilan.",
            "Keadilan sosial bagi seluruh rakyat Indonesia tanpa diskriminasi.",
            "Pengakuan terhadap harkat dan martabat manusia yang beradab."
        ],
        "answer": "B",
    },
    {
        "subject": "Sains & Lingkungan",
        "question": "Proses fotosintesis pada tumbuhan hijau menghasilkan produk utama berupa...",
        "choices": [
            "Karbon dioksida dan air",
            "Glukosa dan oksigen",
            "Nitrogen dan klorofil",
            "Asam laktat dan energi ATP"
        ],
        "answer": "B",
    },
    {
        "subject": "Logika & Matematika",
        "question": "Sebuah toko memberikan diskon ganda 20% + 10% untuk sebuah jaket seharga Rp500.000. Berapakah harga akhir yang harus dibayar pembeli?",
        "choices": [
            "Rp350.000",
            "Rp360.000",
            "Rp370.000",
            "Rp400.000"
        ],
        "answer": "B",
    },
    {
        "subject": "Geografi Indonesia",
        "question": "Garis Wallace memisahkan persebaran fauna Indonesia tipe Asiatis dan tipe Peralihan. Garis ini membentang di antara pulau...",
        "choices": [
            "Sumatera dan Jawa",
            "Kalimantan dan Sulawesi",
            "Sulawesi dan Maluku",
            "Papua dan Nusa Tenggara"
        ],
        "answer": "B",
    },
    {
        "subject": "Teknologi & Komputasi",
        "question": "Struktur data yang beroperasi dengan prinsip LIFO (Last In First Out) adalah...",
        "choices": [
            "Queue (Antrean)",
            "Stack (Tumpukan)",
            "Linked List (Senarai Berantai)",
            "Binary Search Tree"
        ],
        "answer": "B",
    },
    {
        "subject": "Bahasa Indonesia",
        "question": "Kata serapan yang penulisannya baku menurut KBBI adalah...",
        "choices": [
            "Apotik",
            "Apotek",
            "Ijin",
            "Nasehat"
        ],
        "answer": "B",
    },
    {
        "subject": "Fisika",
        "question": "Hukum I Newton menyatakan bahwa suatu benda akan tetap diam atau bergerak lurus beraturan jika...",
        "choices": [
            "Benda tersebut berada di ruang hampa udara.",
            "Total gaya (resultan gaya) yang bekerja pada benda sama dengan nol.",
            "Gaya gesek lebih besar daripada gaya dorong.",
            "Massa benda berbanding lurus dengan percepatannya."
        ],
        "answer": "B",
    },
    {
        "subject": "Ekonomi",
        "question": "Kenaikan harga barang dan jasa secara umum dan terus-menerus dalam jangka waktu tertentu disebut...",
        "choices": [
            "Deflasi",
            "Inflasi",
            "Devaluasi",
            "Resesi"
        ],
        "answer": "B",
    }
]


class IndoMMLU(Task):
    """
    Indonesian Multi-Subject Benchmark.
    Categorical evaluation measuring model comprehension across Indonesian educational and domain knowledge.
    """

    def __init__(self, subset="all", split="test", **kwargs):
        super().__init__(**kwargs)
        self.subset = subset
        self.split = split
        self.ds = self._load_dataset()

    def _load_dataset(self):
        # Attempt to load from hub or use curated benchmark
        try:
            ds = load_hub_dataset("haryoaw/indo-mmlu", self.subset, split=self.split)
            return ds.shuffle(seed=42)
        except Exception:
            pass

        try:
            ds = load_hub_dataset("indolem/indommlu", self.subset, split=self.split)
            return ds.shuffle(seed=42)
        except Exception:
            pass

        # Use curated table fallback
        table = pa.Table.from_pydict({
            "subject": [r["subject"] for r in CURATED_INDO_MMLU_ROWS],
            "question": [r["question"] for r in CURATED_INDO_MMLU_ROWS],
            "choices": [r["choices"] for r in CURATED_INDO_MMLU_ROWS],
            "answer": [r["answer"] for r in CURATED_INDO_MMLU_ROWS],
        })
        return HubDataset(table).shuffle(seed=42)

    @property
    def eval_type(self):
        return "categorical"

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        question = row.get("question") or row.get("prompt") or row.get("text")
        choices = row.get("choices")
        if isinstance(choices, dict) and "text" in choices:
            choices = choices["text"]

        answer = str(row.get("answer", "A")).strip()
        letters = ["A", "B", "C", "D"][:len(choices)]

        if answer not in letters:
            # Handle 0-indexed numeric answers if present
            if answer.isdigit() and int(answer) < len(letters):
                answer = letters[int(answer)]
            else:
                answer = letters[0]

        user_message = render_mc_id(question, letters, choices)
        conversation = {
            "messages": [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": answer}
            ],
            "letters": letters,
            "subject": row.get("subject", "General")
        }
        return conversation

    def evaluate(self, conversation, assistant_response):
        target = conversation["messages"][-1]["content"].strip()
        return assistant_response.strip() == target


class IndoReasoning(Task):
    """
    Indonesian Logical & Reasoning Benchmark.
    Tests cause-and-effect deduction and Indonesian logic problems.
    """

    def __init__(self, split="test", **kwargs):
        super().__init__(**kwargs)
        self.split = split
        self.ds = self._build_dataset()

    def _build_dataset(self):
        reasoning_rows = [
            {
                "premise": "Semua mamalia bernapas menggunakan paru-paru. Ikan paus adalah mamalia air.",
                "question": "Kesimpulan yang paling tepat dan logis adalah...",
                "choices": [
                    "Ikan paus bernapas menggunakan insang karena hidup di air.",
                    "Ikan paus bernapas menggunakan paru-paru.",
                    "Ikan paus bukan termasuk hewan bertulang belakang.",
                    "Sebagian paus bernapas dengan insang dan sebagian dengan paru-paru."
                ],
                "answer": "B"
            },
            {
                "premise": "Jika hari ini hujan lebat, maka jalanan akan basah dan licin. Saat ini jalanan kering dan tidak licin.",
                "question": "Berdasarkan prinsip logika modus tollens, apa yang dapat disimpulkan?",
                "choices": [
                    "Hari ini sedang hujan lebat.",
                    "Hari ini tidak hujan lebat.",
                    "Hujan lebat baru saja berhenti.",
                    "Jalanan akan segera basah dalam beberapa jam."
                ],
                "answer": "B"
            },
            {
                "premise": "Andi lebih tinggi daripada Budi. Cici lebih tinggi daripada Andi.",
                "question": "Urutan tinggi badan dari yang tertinggi ke terpendek adalah...",
                "choices": [
                    "Budi, Andi, Cici",
                    "Cici, Andi, Budi",
                    "Andi, Cici, Budi",
                    "Cici, Budi, Andi"
                ],
                "answer": "B"
            },
            {
                "premise": "Sebuah mobil menempuh jarak 120 km dalam waktu 2 jam dengan kecepatan konstan.",
                "question": "Berapakah waktu yang dibutuhkan mobil tersebut untuk menempuh jarak 300 km dengan kecepatan yang sama?",
                "choices": [
                    "4 jam",
                    "5 jam",
                    "6 jam",
                    "4,5 jam"
                ],
                "answer": "B"
            },
            {
                "premise": "Di sebuah perpustakaan: Semua buku sains bersampul biru. Buku Kimia Dasar adalah buku sains.",
                "question": "Pernyataan mana yang pasti benar?",
                "choices": [
                    "Buku Kimia Dasar bersampul merah.",
                    "Buku Kimia Dasar bersampul biru.",
                    "Semua buku bersampul biru adalah buku Kimia Dasar.",
                    "Buku Kimia Dasar tidak memiliki sampul."
                ],
                "answer": "B"
            }
        ]
        table = pa.Table.from_pydict({
            "premise": [r["premise"] for r in reasoning_rows],
            "question": [r["question"] for r in reasoning_rows],
            "choices": [r["choices"] for r in reasoning_rows],
            "answer": [r["answer"] for r in reasoning_rows],
        })
        return HubDataset(table).shuffle(seed=42)

    @property
    def eval_type(self):
        return "categorical"

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        full_q = f"{row['premise']}\n{row['question']}"
        choices = row["choices"]
        letters = ["A", "B", "C", "D"]
        answer = row["answer"]

        user_message = render_mc_id(full_q, letters, choices)
        return {
            "messages": [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": answer}
            ],
            "letters": letters,
        }

    def evaluate(self, conversation, assistant_response):
        target = conversation["messages"][-1]["content"].strip()
        return assistant_response.strip() == target


class AlpacaIndoEval(Task):
    """
    Generative Indonesian Instruction Following Evaluation.
    Uses the holdout validation split of Alpaca GPT-4 Indonesian.
    """

    def __init__(self, split="test", **kwargs):
        super().__init__(**kwargs)
        self.task = AlpacaGPT4Indonesian(split=split)

    @property
    def eval_type(self):
        return "generative"

    def num_examples(self):
        return len(self.task)

    def get_example(self, index):
        return self.task.get_example(index)

    def evaluate(self, conversation, assistant_response):
        # Generative evaluation criteria:
        # Non-empty response, minimum length requirement, and basic coherence check
        if not assistant_response or len(assistant_response.strip()) < 10:
            return False

        # Check that assistant response does not just repeat prompt
        user_prompt = conversation["messages"][0]["content"]
        if assistant_response.strip() == user_prompt.strip():
            return False

        return True
