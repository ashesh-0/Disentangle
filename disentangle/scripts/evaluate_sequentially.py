import argparse

from disentangle.scripts.evaluate import save_hardcoded_ckpt_evaluations_to_file

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--normalized_ssim', action='store_true')
    parser.add_argument('--save_prediction', action='store_true')
    parser.add_argument('--save_prediction_factor', type=float, default=1.0)

    parser.add_argument('--mmse_count', type=int, default=1)
    parser.add_argument('--start_k', type=int, default=0)
    parser.add_argument('--end_k', type=int, default=1000)
    parser.add_argument('--grid_size', type=int, default=None)
    parser.add_argument('--preserve_older_prediction', action='store_true')
    parser.add_argument('--skip_highsnr', action='store_true')

    args = parser.parse_args()
    if args.preserve_older_prediction:
        assert args.save_prediction, 'Preserving older prediction does not make sense without saving the prediction'

    print('Evaluating between', args.start_k, args.end_k)
    for i in range(args.start_k, args.end_k):
        print('')
        print('##################################')
        print(f'Predicting {i}th frame')
        print('##################################')
        output_stats, pred_unnorm = save_hardcoded_ckpt_evaluations_to_file(
            normalized_ssim=args.normalized_ssim,
            save_prediction=args.save_prediction,
            mmse_count=args.mmse_count,
            predict_kth_frame=i,
            grid_size=args.grid_size,
            overwrite_saved_predictions=not args.preserve_older_prediction,
            save_prediction_factor=args.save_prediction_factor,
            skip_highsnr=args.skip_highsnr)
        if output_stats is None:
            break
