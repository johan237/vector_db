class Schema:
    def __init__(self, required: dict, optional: dict = None):
        self.required = required
        self.optional = optional or {}

        # check overlap
        overlap = set(self.required) & set(self.optional)
        if overlap:
            raise ValueError(f"Fields {overlap} cannot be both required and optional.")

    def validate(self, data: dict):
        # check required fields
        for k in self.required:
            if k not in data:
                raise ValueError(f"Missing required field '{k}'")
        return True

    def all_fields(self):
        """Return all fields (required + optional)."""
        return {**self.required, **self.optional}
