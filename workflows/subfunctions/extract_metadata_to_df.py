import pandas as pd
import re
def extract_metadata_to_df(imgList, imageFileRegEx):
    """
    Extract metadata for each file in imgList using imageFileRegEx and return as a DataFrame.
    """
    rcfpIdx_rows = []
    for img in imgList:
        match = imageFileRegEx.search(img)
        if match:
            rcfpIdx_rows.append({
                'raw': int(match.group('raw')),
                'col': int(match.group('col')),
                'field': int(match.group('field')),
                'zposition': int(match.group('zposition')),
                'channel': int(match.group('channel')[2:]),  # Extract channel number
                'filename': img
            })
        else:
            print(f"Filename '{img}' does not match the expected format.")
    return pd.DataFrame(rcfpIdx_rows)