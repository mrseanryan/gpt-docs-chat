import os
from cornsnake import util_dir, util_file, util_json, util_list


def check_if_new_data(data_dir):
    """
    Take list of docs and their timestamps, and compare it to previously saved list (if it exists).
    """
    files = util_dir.find_files_recursively(data_dir)
    files_and_dates = {}
    for file in files:
        files_and_dates[file] = util_file.get_modified_date(file).strftime(
            "%m/%d/%Y, %H:%M:%S"
        )

    has_new_data = True

    PATH_TO_DATA_CHECKFILE = "./data.checkfile.json"
    if os.path.exists(PATH_TO_DATA_CHECKFILE):
        existing_files_and_dates = util_json.read_from_json_file(PATH_TO_DATA_CHECKFILE)
        if len(existing_files_and_dates.keys()) == len(files_and_dates.keys()):
            intersection = util_list.intersecting(
                files_and_dates.keys(), existing_files_and_dates.keys()
            )
            # for key in files_and_dates.keys():
            #     if key in existing_files_and_dates.keys():
            #         intersection.append(key)
            if len(intersection) == len(existing_files_and_dates.keys()):
                has_new_data = False
                for existing_file in existing_files_and_dates:
                    if (
                        existing_files_and_dates[existing_file]
                        != files_and_dates[existing_file]
                    ):
                        has_new_data = True
                        break

    util_json.write_to_json_file(files_and_dates, PATH_TO_DATA_CHECKFILE)
    return has_new_data
