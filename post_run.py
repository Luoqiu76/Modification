# "analysis": {
#         "baseline_pre": {
#             "win": 2,
#             "lose": 0,
#             "tie": 0,
#             "win_rate": 0.9950248756218907,
#             "tie_rate": 0.0,
#             "win_tie_rate": 0.9950248756218907,
#             "errors": [],
#             "total": 2,
#             "judge_error": 0,
#             "model_response_error": 0
#         },
#         "ours_pre": {
#             "win": 2,
#             "lose": 0,
#             "tie": 0,
#             "win_rate": 0.9950248756218907,
#             "tie_rate": 0.0,
#             "win_tie_rate": 0.9950248756218907,
#             "errors": [],
#             "total": 2,
#             "judge_error": 0,
#             "model_response_error": 0
#         },
#         "total": {
#             "win": 4,
#             "lose": 0,
#             "tie": 0,
#             "win_rate": 0.9975062344139651,
#             "tie_rate": 0.0,
#             "win_tie_rate": 0.9975062344139651
#         },
#         "ours_time": 32.428521275520325,
#         "baseline_time": 8.042502403259277,
#         "ours_baseline_time_ratio": 4.032143187470911
#     }
from argparse import ArgumentParser
import os
import json

def main(args):
    final_results = {}
    for dataset_name in os.listdir(args.work_dir):
        if not os.path.isdir(os.path.join(args.work_dir, dataset_name)):
            continue
        dataset_result_dir = os.path.join(args.work_dir, dataset_name)
        with open(os.path.join(dataset_result_dir, "eval_output.json"), "r", encoding="utf-8") as f:
            eval_output = json.load(f)
        final_results[dataset_name] = eval_output['analysis']
    
    sum_analysis = {
        'baseline_pre': {
            'win': 0,
            'lose': 0,
            'tie': 0,
            'win_rate': 0,
            'tie_rate': 0,
            'win_tie_rate': 0,
            'total': 0,
            'judge_error': 0,
            'model_response_error': 0
        },
        'ours_pre': {
            'win': 0,
            'lose': 0,
            'tie': 0,
            'win_rate': 0,
            'tie_rate': 0,
            'win_tie_rate': 0,
            'total': 0,
            'judge_error': 0,
            'model_response_error': 0
        },
        'total': {
            'win': 0,
            'lose': 0,
            'tie': 0,
            'win_rate': 0,
            'tie_rate': 0,
            'win_tie_rate': 0
            },
        'ours_time': 0,
        'baseline_time': 0,
        'ours_baseline_time_ratio': 0
    }
    for dataset_name, analysis in final_results.items():
        for key in ['baseline_pre', 'ours_pre', 'total']:
            for subkey in ['win', 'lose', 'tie', 'win_rate', 'tie_rate', 'win_tie_rate', 'total', 'judge_error', 'model_response_error']:
                if key == 'total' and subkey in ['total', 'judge_error', 'model_response_error']:
                    continue
                sum_analysis[key][subkey] += analysis[key][subkey]
        for key in ['ours_time', 'baseline_time', 'ours_baseline_time_ratio']:
            sum_analysis[key] += analysis[key]
    final_results['sum'] = sum_analysis
    avg_analysis = sum_analysis.copy()
    for key in ['baseline_pre', 'ours_pre', 'total']:
        for subkey in ['win', 'lose', 'tie', 'win_rate', 'tie_rate', 'win_tie_rate', 'total', 'judge_error', 'model_response_error']:
            if key == 'total' and subkey in ['total', 'judge_error', 'model_response_error']:
                continue
            avg_analysis[key][subkey] /= len(final_results) - 1
        
        for key in ['ours_time', 'baseline_time', 'ours_baseline_time_ratio']:
            avg_analysis[key] /= len(final_results) - 1
    final_results['avg'] = avg_analysis

    with open(os.path.join(args.work_dir, "results_summary.json"), "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=4)
    
    dataset_name = os.listdir(args.work_dir)[0]

    with open(os.path.join(args.work_dir, dataset_name, "run_config.json"), "r", encoding="utf-8") as f:
        run_config = json.load(f)
    
    with open(os.path.join(args.work_dir, dataset_name, "eval_config.json"), "r", encoding="utf-8") as f:
        eval_config = json.load(f)

    with open(os.path.join(args.work_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=4)

    with open(os.path.join(args.work_dir, "eval_config.json"), "w", encoding="utf-8") as f:
        json.dump(eval_config, f, indent=4)

    


    

    



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--work_dir", type=str, help="Path to the save working directory")
    args = parser.parse_args()
    main(args)
