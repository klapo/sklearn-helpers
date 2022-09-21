import yaml


def assign_mbw_type(ds, path_to_yml):
    """Assigns the mbw_type to performance model data.
    """
    # Grab the mbw type from the performance model parameter file.
    with open(path_to_yml, "r") as c:
        params = yaml.safe_load(c)
    
    manh = params['mbw']['manh']
    manh = list(manh.values())

    curvi = params['mbw']['curvi']
    curvi = list(curvi.values())
    
    hugeV = [c for c in curvi if 'hugeVertex' in c]
    curvi = [c for c in curvi if 'hugeVertex' not in c]
    
    # Assign the mbw type.
    ds['mbw_type'] = ('job', [None] * len(ds.job))
    for j in ds.job:
        mbw = ds.sel(job=j).mbw.values
        # Check against the performance model parameters file.
        if mbw in manh:
            mbw_type = 'manh'
        elif mbw in curvi:
            mbw_type = 'curv'
        elif mbw in hugeV:
            mbw_type = 'hugeV'

        # The mbw is not in the performance model parameters file,
        # infer based on the mbw name.
        elif ds.sel(job=j).mbw_version.isin(['2.1', '2.2']):
            if 'curv' in str(ds.sel(job=j).mbw.values):
                mbw_type = 'curv'
            elif 'manh' in str(ds.sel(job=j).mbw.values):
                mbw_type = 'manh'
            elif 'hugeVertex' in str(ds.sel(job=j).mbw.values):
                mbw_type = 'hugeV'
        
        elif ds.sel(job=j).mbw_version.isin(['1.2']):
            # All 1.2 types were manhattan
            mbw_type = 'manh'
        
        else:
            mbw_type = 'indeterminate'
        
        ds['mbw_type'].loc[dict(job=j)] = mbw_type
        
    return ds