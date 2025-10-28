import json

stance_to_int = {'Agree': 0, 'Disagree': 1, 'Discuss': 2, 'Unrelated': 3, \
                 'agree': 0, 'disagree': 1, 'discuss': 2, 'unrelated': 3, \
                 'other': 2} # some datasets merge discuss and unrelated

class AraStanceReader:
    """A class to read and process the AraStance dataset from a JSONL file.
    This dataset contains claims, articles, stances, and indices linking articles to claims.
    
    Args:
        path (str): Path to the dataset directory containing the JSONL file.
    
    Attributes:
        claims (list): List of claims.
        articles (list): List of articles.
        stances (list): List of stances corresponding to the claim/article pairs.
        article_claim (list): Indices linking each article to its corresponding claim.
    """
    def __init__(self, path):
        self.claims, self.articles, self.stances, self.article_claim = self.read(path)
        
    def read(self, path):
        """Reads the dataset from a JSONL file.
        
        Args:
            path (str): Path to the JSONL file containing the dataset.
        
        Returns:
            tuple: A tuple containing lists of claims, articles, stances, and article_claim indices.
        """
        claims, articles, stances = [], [], []
        article_claim = []
        with open(path, 'r', encoding='utf-8') as file:
            for idx, line in enumerate(file):
                instance = json.loads(line)
                claims.append(instance['claim'])
                articles.extend(instance['article'])
                article_claim.extend([idx] * len(instance['article']))
                stances.extend(instance['stance'])
            assert len(stances) == len(articles) == len(article_claim)
            return claims, articles, stances, article_claim
        
    def __len__(self):
        """Returns the number of instances in the dataset."""
        return len(self.stances)
    
    def __getitem__(self, idx):
        """Returns the claim, article, and stance for a given index."""
        return (self.claims[self.article_claim[idx]], 
                self.articles[idx], 
                self.stances[idx])