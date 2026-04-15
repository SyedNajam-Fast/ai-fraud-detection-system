from __future__ import annotations

from src.db import get_or_create_user, insert_transaction


def insert_sample_transaction() -> int:
	"""Insert one deterministic sample transaction for the end-to-end demo."""
	user_id = get_or_create_user(
		name="Ayesha Khan",
		email="ayesha.khan@example.com",
		card_number="4111111111111111",
	)
	return insert_transaction(
		user_id=user_id,
		amount=12450.75,
		time=23,
		location="Lahore",
		merchant="electronics_store",
	)
