# from AIToolbox.AWS.DataAccess import SQuAD2DatasetFetcher
#
# df = SQuAD2DatasetFetcher(bucket_name='dataset-store',
#                           local_dataset_folder_path='~/PycharmProjects/AIToolbox/ToolboxUseScripts/AWS')
#
# df.fetch_dataset()


from AIToolbox.NLP.Dataset import SQuAD2, HotpotQA, QAngaroo, CNNDailyMail

path = '/Users/markovidoni/PycharmProjects/MemoryNet/test_data_dir'

SQuAD2.get_dataset_local_copy(path)
HotpotQA.get_dataset_local_copy(path)
QAngaroo.get_dataset_local_copy(path)
CNNDailyMail.get_preproc_dataset_local_copy(path, 'abisee')

CNNDailyMail.get_preproc_dataset_local_copy(path, 'danqi')

