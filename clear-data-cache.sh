set -e

function delete_dir()
{
    if [ -d $1 ]; then
    rm -rf $1
    fi
}

function delete_file()
{
    if [ -f $1 ]; then
    rm $1
    fi
}

delete_file data.checkfile.json
delete_dir storage_summary
delete_dir storage_vector
