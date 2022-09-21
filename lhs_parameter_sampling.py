import yaml
import numpy as np
import pandas as pd
import os
from scipy.stats import qmc
import copy
import ast


def param_generator_lhs(
    num_samples,
    num_iterations,
    param_file_path,
    df_previous_params=None,
    conditional_vars=None,
    vars_to_ignore=None,
    weights=None,
    batch_num=None,
):
    if batch_num is None:
        batch_num = 0
    
    if weights is None:
        weights = {}
    
    if df_previous_params is None:
        df_previous_params = pd.DataFrame()
    
    batch_runs = {}
    
    with open(param_file_path, "r") as c:
        params = yaml.safe_load(c)
    
    # These variables are conditioned on the selection of another
    # variable.
    if conditional_vars is not None:
        pset = set(list(params.keys()))
        glist = []
        gset = [
            glist.extend(list(conditional_vars[k]))
            for k in conditional_vars.keys()
        ]
        gset = set(glist)
        # Remove them from the list of parameters to consider
        params_to_use = pset.symmetric_difference(gset)
    else:
        conditional_vars = {}
        params_to_use = set(list(params.keys()))

    if vars_to_ignore is None:
        vars_to_ignore = []

    # MBW variables require special handling
    mbw_vars = ['mbw', 'chip_window', 'new_file_name',
                'mbw_vertex_density_mean']
    
    # The list of parameters to sample, not including parameters
    # that are grouped, ignored, or mbw dependent
    params_to_use = params_to_use.symmetric_difference(set(mbw_vars))
    params_to_use = params_to_use - set(vars_to_ignore)
    
    # LHS sampling
    # Add a dimension for the vertex density
    num_dimensions = len(params_to_use) + 1
    lhs_engine = qmc.LatinHypercube(d=num_dimensions)
    lhs_sample = []
    for n in range(num_iterations):
        lsamp = lhs_engine.random(n=num_samples)
        lhs_sample.append(lsamp)
    # The array shape is iterations, samples, data dimensions
    lhs_sample = np.array(lhs_sample)
    lhs_digitize = np.ones_like(lhs_sample) * np.nan
    
    # Digitize the LHS samples to the parameter indices
    for nparam, p in enumerate(params_to_use):
        if p in weights:
            bins = weights[p]
        else:
            num_options = len(params[p])
            bins = np.arange(0, 1, 1 / num_options)
        lhs_digitize[:, :, nparam] = np.digitize(
            lhs_sample[:, :, nparam],
            bins=bins,
            right=True
        )
    # And now mbw vertex density (just an estimate for
    # evaluating unique jobs)
    lhs_digitize[:, :, -1] = np.digitize(
        lhs_sample[:, :, -1],
        bins=np.arange(0, 1, 1 / 10),
        right=True
    )
    # For indexing we need to subtract one
    lhs_digitize = lhs_digitize - 1
    lhs_digitize = lhs_digitize.astype(int)
    
    # Assign the parameters
    for iter_counter in range(num_iterations):
        bname = 'batch {}'.format(iter_counter + batch_num)
        batch_runs[bname] = []

        for sample_counter in range(num_samples):
            run_dict = {}

            for nparam, p in enumerate(params_to_use):
                selection = lhs_digitize[iter_counter, sample_counter, nparam]

                if p == 'aad_ilf':
                    run_dict['aad'] = params[p][selection][0]
                    run_dict['ilf'] = params[p][selection][1]

                elif p == 'mbw_type':
                    run_dict[p] = params[p][selection]

                    # The last dimension of the lhs sampler always corresponds
                    # to the mbw vertex density
                    num_mbw_vd_options = len(
                        params['mbw_vertex_density_mean'][run_dict[p]])
                    selection = np.digitize(
                        lhs_sample[iter_counter, sample_counter, -1],
                        bins=np.arange(0, 1, 1 / num_mbw_vd_options)
                    ) - 1

                    mbw_key = params['mbw_vertex_density_mean'][run_dict[p]][selection]
                    run_dict['mbw_vertex_density_mean'] = mbw_key
                    run_dict['mbw'] = params['mbw'][run_dict[p]][mbw_key]
                    run_dict['mbw_root'] = '/'.join(params['mbw'][run_dict[p]][mbw_key].split('/')[0:4])
                    run_dict['mbw_path'] = '/'.join(params['mbw'][run_dict[p]][mbw_key].split('/')[4:])
                    run_dict['chip_window'] = tuple(
                        params['chip_window'][run_dict[p]][mbw_key])
                    run_dict['new_file_name'] = params['new_file_name'][run_dict[p]][mbw_key]

                elif p in conditional_vars.keys():
                    if params[p][selection]:
                        run_dict[p] = params[p][selection]
                        # The grouped variables are randomly selected. Ideally
                        # this would somehow include a stricter LHS sample, but
                        # it should be good enough.
                        for gv in conditional_vars[p]:
                            low = 0
                            high = len(params[gv])
                            selection = np.random.randint(
                                low, high, size=None, dtype=int)
                            run_dict[gv] = params[gv][selection]

                    else:
                        run_dict[p] = params[p][selection]
                        for gv in conditional_vars[p]:
                            run_dict[gv] = 0

                else:
                    run_dict[p] = params[p][selection]
            batch_runs[bname].append(run_dict)

    # Missing jobs should be filled with a random, unique sample
    df_runs = runs_df_transformer(batch_runs)
    duplicate_index, unique_params_list = runs_df_duplicates(
        params_to_use,
        pd.concat([df_previous_params, df_runs]),
        conditional_vars=conditional_vars
    )
    # Reference the duplicate jobs to the newly generated samples
    if any(duplicate_index < df_previous_params.index.size):
        print('Previous jobdecks had non-unique values. These are being dropped.')
        duplicate_index = duplicate_index[duplicate_index > df_previous_params.index.size]
    duplicate_index = duplicate_index - df_previous_params.index.size
    
    # Replace the non-unique values with new samples. New samples are forced to be unique
    # through a brute force approach.
    if duplicate_index.size > 0:
        print(('Filling duplicated runs using a random sampling strategy. This should happen '
               'with vanishing rarity when the parameter set is much larger than the number '
               'of samples. Re-running the LHS may solve the duplicate issues.')
        )
        
        # Generate randomly sampled parameters to fill the duplicated values.
        missing_runs = duplicate_index.size
        df_fill = param_generator_random(
            params,
            params_to_use,
            missing_runs,
            df_runs.drop(duplicate_index),
            conditional_vars=conditional_vars,
        )

        # Fill the duplicate values with the unique values.
        mask = df_runs.columns == 'batch'
        for n in range(len(duplicate_index)):
            bname = df_runs.iloc[duplicate_index[n]].batch
            df_fill.iloc[n, mask] = bname
            df_runs.iloc[duplicate_index[n]] = df_fill.iloc[n]

    return df_runs, unique_params_list


def runs_df_transformer(all_runs):
    # Turn the hierarchical dictionary into a dataframe with multiindex.
    df = pd.DataFrame.from_dict(all_runs, orient="index").stack().to_frame()
    # And break out each list in the dictionary into the columns.
    df = pd.DataFrame(df[0].values.tolist(), index=df.index)
    
    # Name the multiindex and unwrap it into columns
    df.index.set_names(['batch', 'sample number'], inplace=True)
    df = df.reset_index()
    
    return df

    
def runs_df_duplicates(
    params_to_use,
    df,
    conditional_vars=None,
    keep='first',
    generate_unique_list=True
):
    unique_params = copy.deepcopy(params_to_use)
    if generate_unique_list:
        unique_params.update(('mbw_vertex_density_mean', 'aad', 'ilf'))
        unique_params.update(conditional_vars)
        unique_params.remove('aad_ilf')

    unique_params_list = list(unique_params)    
    duplicate_index = np.flatnonzero(df[unique_params_list].duplicated(keep=keep).values)

    return duplicate_index, unique_params_list


def param_generator_random(params, params_to_use, num_runs_to_parameterize, df_runs, conditional_vars=None):

    missing_runs = num_runs_to_parameterize
    bname = 'fill'
    df_fill = pd.DataFrame()
    fill_run = {}
    
    while missing_runs > 0:
        fill_run[bname] = []
        run_dict = {}
        for p in params_to_use:
            low = 0
            high = len(params[p])
            selection = np.random.randint(low, high, size=None, dtype=int)
            
            if p == 'aad_ilf':
                run_dict['aad'] = params[p][selection][0]
                run_dict['ilf'] = params[p][selection][1]
                
            elif p == 'mbw_type':
                run_dict[p] = params[p][selection]

                # Select again on the mbw vertex density for the mbw_type
                low = 0
                high = len(params['mbw_vertex_density_mean'][run_dict[p]])
                selection = np.random.randint(low, high, size=None, dtype=int)
                
                mbw_key = params['mbw_vertex_density_mean'][run_dict[p]][selection]
                run_dict['mbw_vertex_density_mean'] = mbw_key
                run_dict['mbw'] = params['mbw'][run_dict[p]][mbw_key]
                run_dict['chip_window'] = tuple(params['chip_window'][run_dict[p]][mbw_key])
                run_dict['new_file_name'] = params['new_file_name'][run_dict[p]][mbw_key]

            elif p in conditional_vars.keys():
                if params[p][selection]:
                    run_dict[p] = params[p][selection]
                    for gv in conditional_vars[p]:
                        low = 0
                        high = len(params[gv])
                        selection = np.random.randint(low, high, size=None, dtype=int)
                        run_dict[gv] = params[gv][selection]

                else:
                    run_dict[p] = params[p][selection]
                    for gv in conditional_vars[p]:
                        run_dict[gv] = 0
            
            else:
                run_dict[p] = params[p][selection]
        
        fill_run[bname].append(run_dict)
        df_new = runs_df_transformer(fill_run)
        
        duplicate_index, unique_params_list = runs_df_duplicates(
            params_to_use,
            pd.concat([df_runs, df_fill, df_new]),
            conditional_vars=conditional_vars
        )

        # We found a unique combination to fill with
        if duplicate_index.size == 0:
            df_fill = pd.concat([df_fill, df_new], ignore_index=True)
            missing_runs = missing_runs - 1

    return df_fill


def parameter_csv_read(csv_param_path):
    df_read = pd.read_csv(
        csv_param_path,
        converters={"chip_window": ast.literal_eval}
    )
    
    return df_read