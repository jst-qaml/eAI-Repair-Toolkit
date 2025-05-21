from repair.core.method import RepairMethod


class DemoMethod(RepairMethod):
    pass

    def localize(self, model, input_neg, output_dir, **kwargs):
        print(f"{self.get_name()}: called localize")

    def optimize(self, model, model_dir, weights, input_neg, input_pos, output_dir, **kwargs):
        print(f"{self.get_name()}: called optimize")

    def evaluate(
        self,
        dataset,
        model_dir,
        target_data,
        target_data_dir,
        positive_inputs,
        positive_inputs_dir,
        output_dir,
        num_runs,
        **kwargs,
    ):
        print(f"{self.get_name()}: called evaluate")
