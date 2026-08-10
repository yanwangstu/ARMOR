import json
from torch.utils.data import Dataset


class RLDataset(Dataset):
    def __init__(self, data_path: str, system_prompt_path: str):
        self.dataset_path = data_path
        self.system_prompt_path = system_prompt_path

        with open(data_path, "r", encoding="utf-8") as f:
            self.raw_data = json.load(f)
        
        with open(system_prompt_path, "r", encoding="utf-8") as f:
            self.system_prompt = f.read()

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        return self.raw_data[idx]

if __name__ == '__main__':
    data_path = '../DatasetConstruction/ReconstructedDataset/SplitInfo/train_rl.json'
    system_prompt_path = '../WarmUp/prompt.txt'
    dataset = RLDataset(data_path, system_prompt_path)
    print(f"dataset length {len(dataset)}")

    from torch.utils.data import DataLoader
    # construct dataloader
    def identity_collate_fn(batch):
        new_batch = {sample["id"]: sample for sample in batch}
        return new_batch
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=identity_collate_fn
    )

    for step, batch in enumerate(dataloader):
        print(f'Step: {step}')
        print("Batch structure (batch_size=2):")
        print(json.dumps(batch, indent=4, default=str))
        break

'''
dataset length 8573
Step: 0
Batch structure (batch_size=2):
{
    "12596_2Wiki_train": {
        "id": "12596_2Wiki_train",
        "data_source": [
            "2WikiMultiHopQA",
            "train",
            12596
        ],
        "main_question": "What is the place of birth of the performer of song Figure It Out (French Montana Song)?",
        "main_answer": "Atlanta",
        "chain_of_thought": [
            {
                "think": "To determine the place of birth of the performer, I first need to identify who performed the song *Figure It Out*.",
                "sub_question": "Who is the performer of the song Figure It Out?",
                "retrieval": [],
                "doc": "\"Figure It Out\" is a single by American rapper French Montana, featuring Kanye West and Nas.",
                "doc_type": "golden",
                "sub_answer": "Figure It Out is performed by Kanye West.",
                "evidence": [
                    "Figure It Out",
                    "performer",
                    "Kanye West"
                ]
            },
            {
                "think": "Now that I know Kanye West is the performer, I can find his place of birth to answer the main question.",
                "sub_question": "Where was Kanye born?",
                "retrieval": [],
                "doc": "Born in Atlanta and raised in Chicago, West first became known as a producer for Roc-A-Fella Records in the early 2000s, producing singles for recording artists such as Jay-Z, Ludacris, and Alicia Keys.",
                "doc_type": "golden",
                "sub_answer": "Kanye's place of birth is Atlanta.",
                "evidence": [
                    "Kanye",
                    "place of birth",
                    "Atlanta"
                ]
            }
        ],
        "extra_info": {
            "subset_split": "train_rl",
            "sub_set_index": 0
        }
    },
    "13404_2Wiki_train": {
        "id": "13404_2Wiki_train",
        "data_source": [
            "2WikiMultiHopQA",
            "train",
            13404
        ],
        "main_question": "Who lived longer, Maureen Daly or Jim Dorgan?",
        "main_answer": "Maureen Daly",
        "chain_of_thought": [
            {
                "think": "To determine who lived longer between Maureen Daly and Jim Dorgan, I first need to find out when Maureen Daly was born to establish her lifespan.",
                "sub_question": "When was Maureen Daly born?",
                "retrieval": [],
                "doc": "Maureen Daly( March 15, 1921 \u2013 September 25, 2006), was an Irish- born American writer best known for her 1942 novel\" Seventeenth Summer\", which she wrote while still in her teens.",
                "doc_type": "golden",
                "sub_answer": "Maureen Daly's date of birth is March 15, 1921.",
                "evidence": [
                    "Maureen Daly",
                    "date of birth",
                    "March 15, 1921"
                ]
            },
            {
                "think": "Now that I know Maureen Daly's birthdate, I need to find her death date to calculate the length of her life.",
                "sub_question": "When did Maureen Daly die?",
                "retrieval": [],
                "doc": "James Anthony Dorgan( 26 November 1930 \u2013 30 January 2003) was an Australian rules footballer who played with South Melbourne in the VFL during the 1950s.",
                "doc_type": "golden",
                "sub_answer": "Maureen Daly's date of death was September 25, 2006.",
                "evidence": [
                    "Maureen Daly",
                    "date of death",
                    "September 25, 2006"
                ]
            }
        ],
        "extra_info": {
            "subset_split": "train_rl",
            "sub_set_index": 1
        }
    }
}
'''