import datetime

from src.policies.base_policy import BasePolicy


class StableBaselinesPolicy(BasePolicy):
    def __init__(self, model):
        # save and load the model as a workaround for creating a copy of the policy
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"tmp_model_{timestamp}.zip"
        model.save(filename)
        self.model = model.__class__.load(filename)
        try:
            os.remove(filename)
        except FileNotFoundError:
            pass

    def get_action(self, obs, deterministic=True):
        a, _ = self.model.predict(obs, deterministic=deterministic)
        return a
