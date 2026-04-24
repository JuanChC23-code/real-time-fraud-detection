def validate_input(transaction: dict) -> None:
    """
    Basic security validation for incoming requests.
    """

    if not isinstance(transaction, dict):
        raise ValueError("Input must be a JSON object")

    if len(transaction) == 0:
        raise ValueError("Empty transaction is not allowed")

    if len(transaction) > 100:
        raise ValueError("Too many input fields (possible abuse)")

    for key, value in transaction.items():
        if not isinstance(key, str):
            raise ValueError("Invalid key type in input")

        if value is None:
            raise ValueError(f"Null value detected for field: {key}")