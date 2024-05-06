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
    parser.add_argument("--keep_best", type=int, default=10,
                        help="Number of proteins with top losses during the generation. The higher, the more are saved.")
    parser.add_argument("--max_mute", type=int, default=30,
                        help="Maximum different AA numbers from the init sequence. (0 for no limit)")
    parser.add_argument("--temperature", type=float, default=8,
                        help="Temperature to decide whether to make a 'bad' mutation. The higher, the harder.")
    parser.add_argument("--temperature_step_size", type=int, default=4000,
                        help="Step size to halve the temperature.")
    parser.add_argument("--heuristic_evolution", type=bool, default=False,
                        help="Whether to use the ESM as a generalist heuristic.")
    parser.add_argument("--activity_w", nargs="+", type=float, default=[-0.1],
                        help="The weight of activity loss.")
    parser.add_argument("--seed", type=int, default=1,
                        help="The random seed.")

    args = parser.parse_args()
    return args


def get_antibody(idx):
    # default 0
    assert idx != 5
    antibodies = [
        'MNQDSRHVEFMNLAMDQARIAYALGEVPIGAVIVKDDEVIACSYNQTEQLQNPTGHAELLVIESASKKLKTRRLLDCTLYVTLEPCAMCAGAIILSRIPIIYFASKDPKAGAVQSLYELLNDKRLNHQCEVHQGFMAKESSELLSSFFRELREGSILKTKELGNNSEA',  # default
        'MTDQDYMQLALAQAAEARAAGEVPVGAVIVKDGEVIATGFNRPISRHDPTHHAEIAALRAAATAVGNYRLPGCTLYVTLEPCVMCAGAMMHARIARVVFGARDPKTGACGSVLDLFANTQLNHHAEVVGGVLADDCGTMLSDFFAERRQQSRTEHS',
        'MSDQAFMRLAMNAAAEAKLVGEVPVGAVVVKDGEVIAVGYNQPIGQHDPTAHAEINALRLAAQKLGNYRLVDCTLYVTLEPCAMCAGAMLHARLARVVFGASDPKTGACGSVMNLFAEKKLNHQTELDGGVLAEECGKMLSSFFAERRQQLRTES',
        'MDDALTIARQALATGDVPVGALVVNPDGIVIGTGFNEREANNDPTAHAEVVAIRNAAQRLQNSRLDGCTLVVTLEPCAMCAGAIAQSRISSLVFGAWDEKAGAVGSVWDVLRDPRSIFKVEVTAGVREQECAQLLKEFFSDK',
        'MTHEDAMRIALEEAALAGAKGDVPVGAVILHNGEVIARRHNEREASNDPTAHAEVLALRDASKLLNSWRLSECTLVVTLEPCVMCAGATQSARIGRLVYGAANFEAGATASLYNVMSDPRLGHNPPVEHGVLAAESAALLKEFFGSKRS',
        'MNQDSRHVEFMNLAMDQARIAYALGEVPIGAVIVKDDEVIACSYNQTEQLQNPTGHAELLVIESASKKLKTRRLLDCTLYVTLEPCAMCAGAIILSRIPIIYFASKDPKAGAVQSLYELLNDKRLNHQCEVHQGFMAKESSELLSSFFRELREGSILKTKELGNNSEA',  # replicated, no use
        'MTGQEDKHFMKLALAQAKAGALAGEVPVGAVVVKNGEVIAVAHNAPLGLKDPTAHAEINALRLAAQNLDNYRLEGCTLYVTLEPCAMCSGAAMHARLSRLVYGAPEPKTGAAGSVLNLFDNLQLNHHTQITGGVLAAECVQELQTFFEMRREQHKASKVGNTA',
        'MVSPQDESFMRLAILQAQKAAACNEVPVGAVLVFDDQVIGQGYNQPIRLHDPSAHAEMMAIREAAKSLENYRIPQSTLYVTLEPCAMCCGAILHARVKRVVFGAADPKTGMAGSVDNLFDLKAINHQTDIEGGVLADECGNLLKEFFKQRRS',
        'MNSTASLDQRFMRMALEQGALAAKNGEVPVGAVVVCDDEVVAVGANAPIGDHDPTAHAEIIALRAAAKTLGNYRLPDCRLYVTLEPCAMCSGAIFHARLSEVIYGAPDPKTGAAGSALDLYSNRSINHQTLIRGGVMADECGQTLQAFFARRRTQQREGSGL',
        'MRLALAQAVSAGDAGEVPVGAIVVKDGEVIGRGANAPISRHDPSAHAEIVALRQAADRLGNYRLNDCTVYVTLEPCAMCSGAMFHARIREVVFGASDPKTGVAGSVTNLYDTPQLNHHTTVRGGVLATECGQLLQDFFAHRRQRGRASDALDSSDAPQ',
        'MREALALAGKAAAVGEVPVGAVVVLDGKIVGRGFNQPISGCDPTAHAEIVALRDAAKTLGNYRLVNASLYVTIEPCSMCAGAIVHARIKRLVYGAVEPKAGVAASQQDFFAQPFLNHRVEVQGGVLEEQCRELLQEFFRSRRLGK',
        'MINPFNDIYFMKQALIEAHKALEEGEVPVGAVVVAGNQIIGKGHNLTEKLSDVTAHAEIQAITAASNFLGAKYLKECTLYVTLEPCSMCAGALYWSQIGKVVYGASDERRGANRFPPGLYHPNTVLIYGVEGEACSSLLKQFFQSKR',
        'MHDDTYFMKAALEQAQIAFDLGEIPIGAVVVWDQKIIARGHNQTEQLKDPTAHAEMIAITAACNQIGSKYLSEATVYVTVEPCLMCTGALYWSKVKHIVFGASDEKNGYQKHTKEQWPFHPKASLTKGIMANECAQLMKDFFSTKR',
        'MSNPVFIEAMRKSLELAAKASQQGDIPVGAVVLNPAGEIVGRGHNTREVDNDPMNHAEIVAMREAANANSSWRLDGHTLVVTLEPCTMCAGAAVQARIGRIVFGAFDDKAGAVGSLWDVVRDRRLPHRPEVVSGVLADECAAILSEFFKTQR',
        'MNEYQHMSFMQLAYEQAEIAYSQGEVPIGAVIVKNNEVIASAYNQTEHHQNPIGHAEILAIERAATILQTRRLTDCTLYVTLEPCAMCAGAIVLSRIPIVYFASKDAKAGAVHSLYELLNDTRLNHQCKIHAGLMHGECSTLLSMFFKQLREGHIAKTHQHRQNREE',
        'MSADEIQGFDWQPDVFYMQSALRCAQKAAAADEVPIGAVIVRNGEIIGRAWNQVEMLKDATAHAEMLAITQAEAAVGDWRLNECDLYVTKEPCPMCAGAIVLSRLRRVVFGCGDPKAGAAGGWINLLQSEPLNHRCEVTGGVLGEESAALLRQFFGKKRAPISES',
        'MSDAEIQGFDWQPDVFYMQSALRCARKAAAADEVPIGAVIVRHGEVIARAWNQVEMLKDATAHAEMLAITQAEAAVGDWRLNECDLYVTKEPCPMCAGAIVLARLRRVVFGCPDPKGGAAGGWINLLQSAPLNHRCEVTSGVLGEESATLLRQFFGKKRAPISES',
    ]
    if idx > 16:
        print('[NOTE] This is not a seed, but a mid-seed!')
        antibodies = [
            'MSNPVFIEAMRKSLELAAKASIQGDIPVGAAVLNPAGEIVARGHTTREVDNDPCNHAEIVAMREAANGNRSVRLDGHTLVHTLEGCTMSAGAVVQARIGRIVFGAFDYKAGAYGSLWDVVRDRRLLHRHEVRSGVLADECAAMLSVFCKTQR',
            'MNEYQHMSFMQLPYEQAEIAYSQGFSPIWAVIVKNNEVIASAYNCTEMCQNPIGHAEIGAISRAHTIHGTVRLTDCTQFVTLEECAMCAGAIVLSRIGISYFASKDAKAGAVHSLVELLNDTRLNHQCKIHAGLMHGECSTLLSMFFKQLREGPIAKTHQHRQNREE',
            'MNEYQHMTFMQGAYEQAEIAYSQGEVPIGAVIVKNNCVIAFAYNQTEHHQRPIGHALIDPIERAATILQTRRLCDITLYVTLEPCWMWAVALVLSRIPIVCFASCDAKAGAVHSLYELLNDTRLKHQCKIHAGLMHGECSTLLSMFFKQLREGVIAKTHQHPQNVEE',
            'MNEYEHMSFMQLAYEQAEIAYSQGEEPIGAVISKNNEVIASYYNQTEHCQNPIGHAEIVAYNRAATIKQTFRLGDCTLYVTLEPCAMCAGDIVLSRIIIVLFASKDAKAGAVQSLYELLNDTRLNHQCKIHAGLMHGECSTWLSMFFKQLREGTIANTHQHRQGREL',
            'MNEYQHMTFMQGAYEQAEIAYSQGEVPIGAVIVKNNCVIAFAYNQTEHHQRPIGHALIDPIERAATILQTRRLCDITLYVTLEPCWMWAVALVLSRIPIVCFASCDAKAGAVHSLYELLNDTRLKHQCKIHAGLMHGECSTLLSMFFKQLREGVIAKTHQHPQNVEE',  # p14_iter19814
            'MNHYQHMSFMQLPYRQAEIAYSQGFSPIAAVIVKNNEVIASAYNNTEMCQNPIGHAEKAAISRAKTCHGTVRLTLCTKFVTLEECAMCAGAIVLSRIGISYFASKDAKAGAEHSLVELLNDTRLNHQCKIHAGLMHDPCSTLTSLFFENLREGPYIKTHQHRQNRPE',  # p18_iter9363
        ]
        idx -= 17
    return antibodies[idx], [[1, len(antibodies[idx])]]

def sample(args):
    antibody, limit_range = get_antibody(args.protein)
    TASK = "fixedseqs_fold"
    save_interval = 1
    conf_nonlinear = ['relu', 'leakyrelu', '2', '3', None][4]
    conf_w = [-1]
    focus = [False, [[7, 150]]][0]
    str_heur = 'Heur_' if args.heuristic_evolution else ''
    str_conf_focus= 'focus-' if focus else ''
    str_conf_w = 'conf' if not conf_nonlinear else conf_nonlinear
    folder_name = f'TadA-{args.protein}_fold-i{args.num_recycles}_I-{args.iteration}_Mute{args.max_mute}_{str_heur}{str_conf_focus}{str_conf_w}{conf_w[0]}_Activity{args.activity_w[0]}_T{args.temperature}-{args.temperature_step_size}_seed{args.seed}'
    path = f'output/{folder_name}/sequences.fasta'
    print(path)
    pdb_dir = f'output/{folder_name}/pdb'
    # pdb_dir = ''
    print(pdb_dir)
    regressor_cfg_path = 'conf/activity_esm_gearnet.yaml'

    # Load hydra config from config.yaml
    with hydra.initialize_config_module(config_module="conf"):
        cfg = hydra.compose(
            config_name="config_TadA",
            overrides=[
                f"task={TASK}",
                f"seed={args.seed}",
                "debug=False",
                f"antibody={antibody}",
                "seq_init_random=False",
                f"+folder_name={folder_name}",
                f"+regressor_cfg_path={regressor_cfg_path}",
                f"+tasks.fixedseqs_fold.max_mute={args.max_mute}",
                f"tasks.fixedseqs_fold.keep_best={args.keep_best}",
                f'tasks.fixedseqs_fold.path={path}',
                f'tasks.fixedseqs_fold.limit_range={limit_range}',
                f'tasks.fixedseqs_fold.pdb_dir={pdb_dir}',
                f"+tasks.fixedseqs_fold.focus={focus}",
                f'tasks.fixedseqs_fold.save_interval={save_interval}',
                f'+tasks.fixedseqs_fold.heuristic_evolution={args.heuristic_evolution}',
                f'tasks.fixedseqs_fold.accept_reject.energy_cfg.conf_w={conf_w}',
                f'tasks.fixedseqs_fold.accept_reject.energy_cfg.falsePOS_w={args.activity_w}',
                f'tasks.fixedseqs_fold.accept_reject.energy_cfg.conf_nonlinear={conf_nonlinear}',
                f'tasks.fixedseqs_fold.accept_reject.energy_cfg.num_recycles={args.num_recycles}',
                f'tasks.fixedseqs_fold.accept_reject.temperature.initial={args.temperature}',
                f'tasks.fixedseqs_fold.accept_reject.temperature.step_size={args.temperature_step_size}',
                f'tasks.fixedseqs_fold.num_iter={args.iteration}'  # DEBUG - use a smaller number of iterations
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