import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Moa parameters.")

    parser.add_argument('--model', type=str, default="Qwen/Qwen2-72B-Instruct", help='Model name')
    parser.add_argument('--reference_models', type=str, nargs='+',
                        default=["default_reference_model1", "default_reference_model2"],
                        help='List of reference models')
    parser.add_argument('--temperature', type=float, default=0.7, help='Temperature for the model')
    parser.add_argument('--max_tokens', type=int, default=512, help='Maximum number of tokens')
    parser.add_argument('--rounds', type=int, default=1, help='Number of rounds')
    parser.add_argument('--multi_turn', type=bool, default=True, help='Enable multi-turn conversation')

    return parser.parse_args()