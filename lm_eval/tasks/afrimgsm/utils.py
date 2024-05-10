import argparse
import yaml

languages = ['eng', 'amh', 'ibo', 'fra', 'sna', 'lin', 'wol', 'ewe', 'lug', 'xho', 'kin', 'twi', 'zul', 'orm', 'yor', 'hau', 'sot', 'swa']

configs = {
    "QUESTION": "Question:",
    "ANSWER": "Step-by-Step Answer:",
    "DIRECT": "Answer:",
    "REGEX": "The answer is (\\-?[0-9\\.\\,]+)"}


def gen_lang_yamls(output_dir: str, overwrite: bool, mode: str) -> None:
    """
    Generate a yaml file for each language.

    :param output_dir: The directory to output the files to.
    :param overwrite: Whether to overwrite files if they already exist.
    """
    err = []
    for lang in languages:
        try:
            if mode == "direct":
                task_name = f"afrimgsm_direct_{lang}"
                yaml_template = "afrimgsm_common_yaml"
            elif mode == "native-cot":
                task_name = f"afrimgsm_native_cot_{lang}"
                yaml_template = "afrimgsm_common_yaml"
            elif mode == "en-cot":
                task_name = f"afrimgsm_en_cot_{lang}"
                yaml_template = "afrimgsm_common_yaml"

            file_name = f"{task_name}.yaml"
            with open(
                f"{output_dir}/{file_name}", "w" if overwrite else "x", encoding="utf8"
            ) as f:
                f.write("# Generated by utils.py\n")
                yaml.dump(
                    {
                        "include": yaml_template,
                        "dataset_name": lang,
                        "task": f"{task_name}"
                    },
                    f,
                    allow_unicode=True,
                    width=float("inf"),
                )
        except FileExistsError:
            err.append(file_name)

    if len(err) > 0:
        raise FileExistsError(
            "Files were not created because they already exist (use --overwrite flag):"
            f" {', '.join(err)}"
        )


def main() -> None:
    """Parse CLI args and generate language-specific yaml files."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--overwrite",
        default=True,
        action="store_true",
        help="Overwrite files if they already exist",
    )
    parser.add_argument(
        "--output-dir", default="./direct", help="Directory to write yaml files to"
    )
    parser.add_argument(
        "--mode",
        default="direct",
        choices=["direct", "native-cot", "en-cot"],
        help="Mode of chain-of-thought",
    )
    args = parser.parse_args()

    gen_lang_yamls(output_dir=args.output_dir, overwrite=args.overwrite, mode=args.mode)


if __name__ == "__main__":
    main()
