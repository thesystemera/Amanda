import os
import sqlite3
import random
import time
import queue
import threading
import asyncio
import numpy as np
from annoy import AnnoyIndex
from sentence_transformers import SentenceTransformer
import config


class VectorService:
    def __init__(self):
        config.custom_print("Lifespan", f"VectorService: loading {config.EMBED_MODEL_NAME} on {config.DEVICE}...")
        self.model = SentenceTransformer(config.EMBED_MODEL_NAME, device=str(config.DEVICE))
        self.model.eval()

        self.indices = {}
        self.shotgun_cache = {}

        for dom in config.DOMAINS:
            idx_path = os.path.join(config.VECTOR_DIR, f"{dom}_embeddings_index.ann")
            self.indices[dom] = self._load_annoy(idx_path)

        per_domain = ", ".join(f"{d}={self.indices[d].get_n_items()}" for d in config.DOMAINS)
        total = sum(i.get_n_items() for i in self.indices.values())
        config.custom_print("Lifespan", f"VectorService: ready ({total} items across {len(config.DOMAINS)} domains — {per_domain}).")

        self._rng = random.Random()
        self._request_queue = queue.Queue()
        self._worker_thread = threading.Thread(target=self._worker, daemon=True, name="vector-worker")
        self._worker_thread.start()

    @staticmethod
    def _load_annoy(path):
        index = AnnoyIndex(config.EMBED_DIM, 'angular')
        if os.path.exists(path):
            try:
                index.load(path)
            except Exception:
                config.custom_print(
                    "Vector",
                    f"Failed to load index {path} (dimension mismatch? run rebuild to regenerate)",
                )
        return index

    def _generate_embedding(self, text: str) -> np.ndarray:
        return self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def _get_db_conn(self, domain: str):
        db_path = os.path.join(config.VECTOR_DIR, f"{domain}_embeddings.db")
        if os.path.exists(db_path):
            return sqlite3.connect(db_path)
        return None

    async def query(self, text, domain, top_n=5, threshold=0.0, force_best=False):
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        self._request_queue.put((text, domain, top_n, threshold, force_best, future))
        return await future

    def _worker(self):
        while True:
            item = self._request_queue.get()
            if item is None:
                break
            text, domain, top_n, threshold, force_best, future = item
            t_total_start = time.perf_counter()

            try:
                if domain not in self.indices or self.indices[domain].get_n_items() == 0:
                    result = []
                    config.custom_print("Vector", f"[{domain}] \"{text}\" | index empty")
                else:
                    t_embed_start = time.perf_counter()
                    query_embedding = self._generate_embedding(text)
                    t_embed_ms = (time.perf_counter() - t_embed_start) * 1000

                    t_search_start = time.perf_counter()
                    item_ids = self.indices[domain].get_nns_by_vector(query_embedding, top_n * 10)
                    t_search_ms = (time.perf_counter() - t_search_start) * 1000

                    results = []
                    best_miss = None
                    now = time.time()
                    conn = self._get_db_conn(domain)
                    if conn:
                        c = conn.cursor()
                        table_name = f"{domain}_embeddings"
                        for item_id in item_ids:
                            c.execute(
                                f"SELECT filename, title, subtitle, embedding FROM {table_name} LIMIT 1 OFFSET ?",
                                (item_id,),
                            )
                            row = c.fetchone()
                            if row:
                                filename, title, subtitle, embedding_bytes = row
                                embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                                norm = np.linalg.norm(embedding)
                                if norm > 0:
                                    embedding = embedding / norm
                                similarity = float(np.dot(query_embedding, embedding))

                                if similarity >= threshold or force_best:
                                    if filename not in self.shotgun_cache or (
                                        now - self.shotgun_cache[filename]
                                    ) >= config.SHOTGUN_COOLDOWN:
                                        results.append((filename, title, subtitle, similarity))

                                if best_miss is None or similarity > best_miss[3]:
                                    best_miss = (filename, title, subtitle, similarity)
                        conn.close()

                    t_total_ms = (time.perf_counter() - t_total_start) * 1000

                    results_str = " ".join(
                        f"{os.path.basename(r[0])}={r[3]:.2f}" for r in results[:top_n]
                    )
                    if not results_str:
                        results_str = "none"

                    if not results:
                        result = []
                        miss_info = ""
                        if best_miss:
                            miss_title = str(best_miss[1]) if best_miss[1] else os.path.basename(best_miss[0])
                            miss_info = f' | BEST=\"{miss_title}\" sim={best_miss[3]:.2f}'
                        config.custom_print(
                            "Vector",
                            f"[{domain}] QUERY=\"{text}\" | embed={t_embed_ms:.0f}ms search={t_search_ms:.0f}ms "
                            f"total={t_total_ms:.0f}ms | no match (threshold={threshold}){miss_info}",
                        )
                    else:
                        results.sort(key=lambda x: x[3], reverse=True)
                        if domain in config.INSTANT_DOMAINS:
                            choice = results[0]
                        else:
                            top_sim = results[0][3]
                            pool = [r for r in results if abs(r[3] - top_sim) < 0.01]
                            choice = self._rng.choice(pool)

                        self.shotgun_cache[choice[0]] = now
                        result = [choice]
                        matched_text = str(choice[1]) if choice[1] else os.path.basename(choice[0])
                        config.custom_print(
                            "Vector",
                            f"[{domain}] QUERY=\"{text}\" | embed={t_embed_ms:.0f}ms search={t_search_ms:.0f}ms "
                            f"total={t_total_ms:.0f}ms | results: {results_str} | chosen: {os.path.basename(choice[0])} "
                            f"sim={choice[3]:.2f} | MATCH=\"{matched_text}\"",
                        )
            except Exception as e:
                config.custom_print("Error", f"VectorService query failed ({domain}): {e}")
                result = []

            if not future.done():
                loop = future.get_loop()
                if loop.is_running():
                    loop.call_soon_threadsafe(future.set_result, result)

    def get_random_meta_tags(self, num_tags=10):
        conn = self._get_db_conn("meta")
        if not conn:
            return ""
        c = conn.cursor()
        c.execute("SELECT title FROM meta_embeddings")
        all_tags = [row[0] for row in c.fetchall() if row[0]]
        conn.close()
        if not all_tags:
            return ""
        random.shuffle(all_tags)
        selected = [all_tags.pop() for _ in range(min(num_tags, len(all_tags)))]
        return (
            "Provide meta-data only when applicable. All emotional meta-data must "
            "be encapsulated. META-DATA examples:\n"
        ) + "\n".join([f"*{tag}*" for tag in selected])
