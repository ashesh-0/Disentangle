import argparse

from disentangle.scripts.evaluate import save_hardcoded_ckpt_evaluations_to_file

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--normalized_ssim', action='store_true')
    parser.add_argument('--save_prediction', action='store_true')
    parser.add_argument('--mmse_count', type=int, default=1)
    parser.add_argument('--start_k', type=int, default=0)
    parser.add_argument('--end_k', type=int, default=1000)
    parser.add_argument('--chunk_k', type=int, default=1)
    parser.add_argument('--full_prediction', action='store_true')

    args = parser.parse_args()
    print('Evaluating between', args.start_k, args.end_k, 'with a step size of ', args.chunk_k)
    for i in range(args.start_k, args.end_k, args.chunk_k):
        print('')
        print('##################################')
        print(f'Predicting {i}th frame')
        print('##################################')
        output_stats, pred_unnorm = save_hardcoded_ckpt_evaluations_to_file(
            normalized_ssim=args.normalized_ssim,
            save_prediction=args.save_prediction,
            mmse_count=args.mmse_count,
            predict_kth_frame=i if args.chunk_k == 1 else list(range(i, min(args.end_k, i + args.chunk_k))),
            full_prediction=args.full_prediction)
        if output_stats is None:
            break
