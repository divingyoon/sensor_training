import unittest

from training.pipelines.train_comparison import _parse_trial_list, _split_indices_by_trial


class FakeDataset:
    def __init__(self, sample_trial_ids: list[str]) -> None:
        self.sample_trial_ids = sample_trial_ids

    def __len__(self) -> int:
        return len(self.sample_trial_ids)


class TrialSplitTest(unittest.TestCase):
    def test_seed_split_does_not_mix_trials_between_train_and_val(self) -> None:
        dataset = FakeDataset(
            [
                "trial_a",
                "trial_a",
                "trial_a",
                "trial_b",
                "trial_b",
                "trial_c",
                "trial_c",
            ]
        )

        split = _split_indices_by_trial(dataset, seed=7)

        train_trials = {dataset.sample_trial_ids[i] for i in split.train_indices}
        val_trials = {dataset.sample_trial_ids[i] for i in split.val_indices}

        self.assertTrue(split.train_indices)
        self.assertTrue(split.val_indices)
        self.assertTrue(train_trials.isdisjoint(val_trials))

    def test_explicit_val_trials_route_only_those_trials_to_val(self) -> None:
        dataset = FakeDataset(
            [
                "trial_a",
                "trial_b",
                "trial_b",
                "trial_c",
                "trial_c",
            ]
        )

        split = _split_indices_by_trial(dataset, seed=7, val_trials=["trial_b"])

        self.assertEqual(split.val_trials, ["trial_b"])
        self.assertEqual({dataset.sample_trial_ids[i] for i in split.val_indices}, {"trial_b"})
        self.assertEqual({dataset.sample_trial_ids[i] for i in split.train_indices}, {"trial_a", "trial_c"})

    def test_explicit_test_trials_are_excluded_from_train_and_val(self) -> None:
        dataset = FakeDataset(
            [
                "trial_a",
                "trial_b",
                "trial_c",
                "trial_d",
            ]
        )

        split = _split_indices_by_trial(dataset, seed=7, val_trials=["trial_b"], test_trials=["trial_d"])

        self.assertEqual(split.test_trials, ["trial_d"])
        self.assertEqual({dataset.sample_trial_ids[i] for i in split.test_indices}, {"trial_d"})
        self.assertEqual({dataset.sample_trial_ids[i] for i in split.train_indices}, {"trial_a", "trial_c"})
        self.assertEqual({dataset.sample_trial_ids[i] for i in split.val_indices}, {"trial_b"})

    def test_too_few_trials_raise_clear_error(self) -> None:
        dataset = FakeDataset(["trial_a", "trial_a"])

        with self.assertRaisesRegex(RuntimeError, "at least 2 distinct trials"):
            _split_indices_by_trial(dataset, seed=7)

    def test_parse_trial_list_accepts_comma_or_space_separated_values(self) -> None:
        self.assertEqual(_parse_trial_list(["trial_a,trial_b", "trial_c"]), ["trial_a", "trial_b", "trial_c"])
        self.assertIsNone(_parse_trial_list(None))


if __name__ == "__main__":
    unittest.main()
