import os
import numpy as np
import esm
import esm.inverse_folding
from esm.inverse_folding.util import extract_coords_from_structure
from biotite.structure.io import pdb
from biotite.structure import filter_backbone
from tqdm import tqdm


def rigid_transform_3D(A, B):
    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # sanity check
    #if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        # print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2,:] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    return R, t


def load(path, verbose):
    '''
        1. del soft chain
        2. change the antibody from chain 'A' to chain 'B'
    '''
    with open(path) as f:
        protein = f.readlines()
    protein = [atom.strip().split() for atom in protein]

    L_all = len(protein)
    soft_chain = list(range(14, 30))

    prev = 0
    for i in range(L_all - 1, -1, -1):
        # only keep the atoms, delete the soft chain
        if (len(protein[i]) < 9) or (int(protein[i][5]) in soft_chain):
            del protein[i]
            continue

        # ckeck atom form validness
        if prev:
            assert prev == len(protein[i])
        prev = len(protein[i])

        # change the antibody to chain 'B'
        if int(protein[i][5]) > 29:
            protein[i][4] = 'B'
    if verbose:
        print(f"[LOG] Load {len(protein)} atoms from {L_all} lines.")
    return protein


def extract_coords(protein, verbose):
    '''
    '''
    N = len(protein)
    coords = []
    backbone = []
    prev_aa = protein[0][5]
    cnt_atom = 0
    sum_atom = 0
    for i in range(N):
        coord = np.array(list(map(float, protein[i][6:9])))
        coords.append(coord)
        if protein[i][5] != prev_aa:
            backbone.append(sum_atom / cnt_atom)
            cnt_atom = 0
            sum_atom = 0
            prev_aa = protein[i][5]
        else:
            cnt_atom += 1
            sum_atom += coord
    backbone.append(sum_atom / cnt_atom)    

    coords = np.array(coords)
    backbone = np.array(backbone)
    if verbose:
        print(f"[LOG] Generate atom's coordinate {backbone.shape}.")
        print(f"[LOG] Generate backbone's coordinate {coords.shape}.")
    return backbone, coords


def change(pdbfile, newfolder, verbose=False):
    protein = load(pdbfile, verbose)
    coords, coords_all = extract_coords(protein, verbose) # [L, xyz]

    # antibody (soft chain) + antigen - soft_chain
    hands = [[44+13-16, 56+13-16], [70+13-16, 79+13-16], [118+13-16, 129+13-16]]

    old_coord = [coords[range(*hand)].mean(0) for hand in hands]
    d01 = np.linalg.norm(old_coord[0]-old_coord[1])
    d02 = np.linalg.norm(old_coord[0]-old_coord[2])

    
    start = np.array([8, 0, 0])
    new_coord = [start]
    new_coord.append(np.array([0, (d01 * d01 - start[0] * start[0])**0.5, 0]))
    new_coord.append(np.array([0, 0, (d02 * d02 - start[0] * start[0])**0.5]))
    if verbose:
        print(f'[LOG] Old coordinate setting: {old_coord}')
        print(f'[LOG] New coordinate setting: {new_coord}')
    try:
        W, B = rigid_transform_3D(np.array(old_coord).T, np.array(new_coord).T)
    except:
        return
    new_coords_all = W @ coords_all.T + B
    for i in range(len(protein)):
        for j in range(3):
            protein[i][6+j] = float(new_coords_all.T[i][j])

    new_path = os.path.join(newfolder, os.path.basename(pdbfile))
    write(new_path, protein, verbose)


def write(path, protein, verbose):
    dir_ = os.path.dirname(path)
    if not os.path.exists(dir_):
        os.makedirs(dir_)

    with open(path, 'w') as f:
        for i in range(len(protein)):
            line = protein[i][0].ljust(6) + \
                protein[i][1].rjust(5) + \
                ' ' * 2 + \
                protein[i][2].ljust(4) + \
                protein[i][3].ljust(4) + \
                protein[i][4].ljust(1) + \
                protein[i][5].rjust(4) + \
                ' ' * 4 + \
                f'{protein[i][6]:>8.3f}' + \
                f'{protein[i][7]:>8.3f}' + \
                f'{protein[i][8]:>8.3f}' + \
                protein[i][9].rjust(6) + \
                protein[i][10].rjust(6) + \
                ' ' * 10 + \
                protein[i][11].rjust(2) + '\n'
            f.write(line)
    if verbose:
        print(f'[LOG] Write {len(protein)} atoms to pdbfile.')


def toy_example():
    a = np.array([0, 0, 1])
    b = np.array([0, 1, 0])
    c = np.array([1, 0, 0])
    z = np.array([0, 2, 0])
    a_prime = np.array([-1, 0, 1])
    b_prime = np.array([0, 0, 0])
    c_prime = np.array([-1, -1, 0])
    z_prime = np.array([1, 0, 0])
    d = np.array([1, 1, 1])
    # d_prime = np.array([0, -1, 1])

    W, B = rigid_transform_3D(np.array([a, b, c]).T, np.array([a_prime, b_prime, c_prime]).T)
    # Transforming the point D to the new coordinate system
    d_prime = W @ np.array([d]).T + B
    print("Position of D' in the new coordinate system:", d_prime.flatten())


if __name__ == '__main__':
    # Example usage
    toy_example()
    for i in tqdm(range(1,2865)):
        pdbfile = f'data/pdb_ESM3_-10_i4_seed0/iter_{i}_0.pdb'
        newfolder = 'data/pdb_ESM3_-10_i4_seed0_changed'
        change(pdbfile, newfolder)