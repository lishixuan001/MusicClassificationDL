# music-cs182 (Python 2.6)

# Set Up

1) Check the req.txt file and make sure that the right dependancies are installed. 

To make the jupyter notebook work:
1) Download the 1.8GB subset from MSD (https://labrosa.ee.columbia.edu/millionsong/pages/getting-dataset#subset)
2) If necessary, unzip the downloaded file.
3) Without modifying the directory name ("MillionSongSubset") or contents of the downloaded data, place the directory in the "./data/MSD" directory in the root. 

4) Cannot get pyechonest. just download the subset here https://labrosa.ee.columbia.edu/millionsong/tasteprofile and put into ./data directory.
You can follow the following code to reduce the data set and store as csv:
```python
def reduce_taste_subset(path='./data/FullEchoNestTasteProfileSubset.txt', 
                        to_path='./data/SmallerEchoNestTasteProfileSubset.csv', downsample=0.1):
    data = pd.read_csv(path, sep="\t", header=None)
    data.columns = ['user', 'song', 'play_count']
    data.astype({'user': np.str, 'song': np.str, 'play_count': np.int32})
    data.sample(frac=downsample).to_csv(to_path, index=False)
```
## Collaborative Filtering
Take a look at this paper: http://dx.doi.org/10.1109/ICDM.2008.22

To install spark: https://blog.sicara.com/get-started-pyspark-jupyter-guide-tutorial-ae2fe84f594f

If there are java errors, usually just downloading the latest Java SDK helps.
