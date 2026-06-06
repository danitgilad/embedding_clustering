from main import build_parser, parse_overrides


def test_parser_has_part_subcommands():
    parser = build_parser()
    args = parser.parse_args(["part-b", "generate", "--n", "5"])
    assert args.part == "part-b" and args.stage == "generate" and args.n == 5


def test_parse_overrides_dotted():
    ov = parse_overrides(["part_b.n_images=42", "seed=1"])
    assert ov == {"part_b.n_images": 42, "seed": 1}
