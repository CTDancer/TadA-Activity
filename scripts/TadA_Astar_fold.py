import os
import time
import argparse
import hydra
from lm_design_fold import Designer


def count_diff(a, b):
    assert len(a) == len(b)
    cnt = sum([1 if a[i] != b[i] else 0 for i in range(len(a))])
    return cnt


def get_args():
    parser = argparse.ArgumentParser(description="AI-Generated Protein")
    parser.add_argument("--protein", type=int, default=0,
                        help="Which protein seed to select.")
    parser.add_argument("--iteration", type=int, default=10000,
                        help="Iterations of the protein evolution.")
    parser.add_argument("--num_recycles", type=int, default=4, choices=[1,2,3,4],
                        help="Number of recycles to fold the protein structure. The higher, the more accurate.")
    parser.add_argument("--keep_best", type=int, default=1500,
                        help="Number of proteins with top losses during the generation. The higher, the more are saved.")
    parser.add_argument("--queue_size", type=int, default=1000,
                        help="Number of protein seeds to hold in a queue. The higher, the more are saved.")
    # parser.add_argument("--max_mute", type=int, default=30,
    #                     help="Maximum different AA numbers from the init sequence. (0 for no limit)")
    parser.add_argument("--conf_threshold", type=float, default=0.75,
                        help="Confidence threshold to decide whether to take the protein.")
    parser.add_argument("--decline_rate", type=float, default=0.999,
                        help="Decline rate for each selection from the queue.")
    parser.add_argument("--heuristic_evolution", type=bool, default=False,
                        help="Whether to use the ESM as a generalist heuristic.")
    # parser.add_argument("--activity_w", nargs="+", type=float, default=[-0.1],
    #                     help="The weight of activity loss.")
    parser.add_argument("--seed", type=int, default=1,
                        help="The random seed.")

    args = parser.parse_args()
    return args


def get_protein(idx):
    pool_dict = {
        14: [
            'MNEYQHMSFMQLAYEQAEIAYSQGEVPIGAVIVKNNEVIASAYNQTEHHQNPIGHAEILAIERAATILQTRRLTDCTLYVTLEPCAMCAGAIVLSRIPIVYFASKDAKAGAVHSLYELLNDTRLNHQCKIHAGLMHGECSTLLSMFFKQLREGHIAKTHQHRQNREE',  # p14
            'MNEYQHMTFMQGAYEQAEIAYSQGEVPIGAVIVKNNCVIAFAYNQTEHHQRPIGHALIDPIERAATILQTRRLCDITLYVTLEPCWMWAVALVLSRIPIVCFASCDAKAGAVHSLYELLNDTRLKHQCKIHAGLMHGECSTLLSMFFKQLREGVIAKTHQHPQNVEE',  # p14_iter19814
            'MNHYQHMSFMQLPYRQAEIAYSQGFSPIAAVIVKNNEVIASAYNNTEMCQNPIGHAEKAAISRAKTCHGTVRLTLCTKFVTLEECAMCAGAIVLSRIGISYFASKDAKAGAEHSLVELLNDTRLNHQCKIHAGLMHDPCSTLTSLFFENLREGPYIKTHQHRQNRPE',  # p18_iter9363
        ]
    }
    pool = pool_dict[idx]
    assert all([len(pool[0]) == len(pool[i]) for i in range(len(pool))])
    return pool, len(pool[0])


def sample(args):
    if args.protein > 0:
        antibody, length = get_protein(args.protein)
    else:
        antibody = '/home/ubuntu/scratch/jingao/Interact/output/TadA-14_fold-i4_I-2000000_B-1500_Heur_focus-Thresh0.75_decline-0.999_conf_seed1'
        length = 167
    TASK = "stage_Astar_fold"
    save_interval = 1
    focus = [False, [[7, length-17]]][1]
    str_heur = 'Heur_' if args.heuristic_evolution else ''
    str_conf_focus= 'focus-' if focus else ''
    folder_name = f'TadA-{args.protein}_fold-i{args.num_recycles}_I-{args.iteration}_B-{args.keep_best}_Q-{args.queue_size}_{str_heur}{str_conf_focus}Thresh{args.conf_threshold}_decline-{args.decline_rate}_conf_seed{args.seed}'
    regressor_cfg_path = 'conf/activity_esm_gearnet.yaml'
    
    # NOTE: Do not change, because the resume function needs the basenames.
    path = f'output/{folder_name}/sequences.fasta'
    print(path)
    best_path = f'output/{folder_name}/best_sequences.fasta'
    print(best_path)
    queue_path = f'output/{folder_name}/queue_sequences.fasta'
    print(queue_path)

    # Load hydra config from config.yaml
    with hydra.initialize_config_module(config_module="conf"):
        cfg = hydra.compose(
            config_name="config_Astar_TadA",
            overrides=[
                f"task={TASK}",
                f"seed={args.seed}",
                "debug=False",
                f"+antibody={antibody}",
                "seq_init_random=False",
                f"+folder_name={folder_name}",
                f"+regressor_cfg_path={regressor_cfg_path}",
                f"tasks.stage_Astar_fold.queue_size={args.queue_size}",
                f"tasks.stage_Astar_fold.keep_best={args.keep_best}",
                f'tasks.stage_Astar_fold.path={path}',
                f'+tasks.stage_Astar_fold.best_path={best_path}',
                f'+tasks.stage_Astar_fold.queue_path={queue_path}',
                f'tasks.stage_Astar_fold.conf_threshold={args.conf_threshold}',
                f'tasks.stage_Astar_fold.decline_rate={args.decline_rate}',
                f'tasks.stage_Astar_fold.limit_range={[[1, length]]}',
                f"+tasks.stage_Astar_fold.focus={focus}",
                f'tasks.stage_Astar_fold.save_interval={save_interval}',
                f'+tasks.stage_Astar_fold.heuristic_evolution={args.heuristic_evolution}',
                f'tasks.stage_Astar_fold.accept_reject.energy_cfg.num_recycles={args.num_recycles}',
                f'tasks.stage_Astar_fold.num_iter={args.iteration}'  # DEBUG - use a smaller number of iterations
            ])

    # Create a designer from configuration
    des = Designer(cfg)

    # Run the designer
    start_time = time.time()
    des.run_from_cfg()
    print("finished after %s hours", (time.time() - start_time) / 3600)
    
    diff = count_diff(des.output_seq, des.init_seqs)
    print(f"Final seq has changed {diff} amino acids. \n {des.output_seq}")

    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    with open(path, 'a') as f:
        f.write(f'>sample_iter{args.iteration}\n')
        f.write(f'{des.output_seq}\n')

    for i in range(args.keep_best):
        best_seq = des.best_seq[i]
        seq = best_seq[2]
        diff = count_diff(seq, des.init_seqs)
        print(f"Best seq (loss={best_seq[1]}) has changed {diff} amino acids, in step {best_seq[0]}. \n {seq}")
        with open(path, 'a') as f:
            f.write(f'>best_iter{best_seq[0]}_loss{round(best_seq[1], 2)}_{best_seq[3]}\n')
            f.write(f'{seq}\n')

if __name__ == '__main__':
    args = get_args()
    sample(args)