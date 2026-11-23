# conf/

Hydra configuration of `generate_tacto_dataset.py` (`@hydra.main(config_path="conf", config_name="digit")`).

Copy from the TACTO repository (https://github.com/facebookresearch/tacto, folder `examples/conf/`):

* `digit.yaml` – sections `tacto` (renderer), `digit` (sensor body URDF) and `object` (indenter body; the
  generator replaces `urdf_path` with `objects/<shape>.urdf`);
* `bg_digit_240_320.jpg` – background image of the DIGIT.

Then add the `experiment` section described in [docs/dataset_preparation.md](../../docs/dataset_preparation.md).
These files are not included here because they belong to the TACTO project.
