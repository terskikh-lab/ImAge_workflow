import pandas as pd
def check_zslice_consistency(rcfpIdx):
    """
    Checks if all (channel, field, col, raw) combinations have the same number of z positions.
    Returns the expected z count if consistent, otherwise raises ValueError and prints problematic images.
    """
    z_counts = rcfpIdx.groupby(['channel', 'field', 'col', 'raw'])['zposition'].nunique()
    if z_counts.nunique() > 1:
        print("Error: Not all (channel, field, col, raw) combinations have the same number of z positions!")
        print(z_counts)
        expected_z = z_counts.mode()[0]
        bad_groups = z_counts[z_counts != expected_z]
        bad_images = []
        for idx, count in bad_groups.items():
            ch, fn, cn, rn = idx
            group_imgs = rcfpIdx[
                (rcfpIdx['channel'] == ch) &
                (rcfpIdx['field'] == fn) &
                (rcfpIdx['col'] == cn) &
                (rcfpIdx['raw'] == rn)
            ]['filename'].tolist()
            bad_images.extend(group_imgs)
            print(f"Problem at channel={ch}, field={fn}, col={cn}, row={rn}: {count} z-slices, files: {group_imgs}")
        print("Images with incorrect z positions are listed above.")
        raise ValueError("Some (channel, field, col, raw) combinations do not have the expected number of z positions. See above for details.")
    else:
        print(f"All (channel, field, col, raw) combinations have {z_counts.iloc[0]} z positions.")
        return z_counts.iloc[0]