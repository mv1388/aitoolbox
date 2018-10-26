from AIToolbox.AWS.DataAccess import SQuAD2DatasetFetcher


df = SQuAD2DatasetFetcher(bucket_name='dataset-store',
                          local_dataset_folder_path='~/PycharmProjects/AIToolbox/ToolboxUseScripts/AWS')

df.fetch_dataset()
