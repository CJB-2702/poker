import random

SUITS = ("Hearts", "Diamonds", "Spades", "Clubs")
RANKS = ("2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A")

suits = {"Hearts": "♥", "Diamonds": "♦", "Spades": "♠", "Clubs": "♣"}
values = {"2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10, "J": 11, "Q": 12, "K": 13, "A": 14}


class Card:
    def __init__(self, rank, suit):
        self.suit = suit
        self.rank = rank
        self.value = values[rank]

    def __str__(self):
        return str(self.rank, self.suit)
    
    def pretty_string(self):
        card_rows=[]
        card_width= 7
        
        rank_spaces= ( card_width-2 ) * " " if self.rank == "10" else ( card_width - 1 ) * " "
        suit_spaces=int(card_width/2+ card_width%2) -1

        card_rows.append("-" *(card_width+2))
        card_rows.append("|"+  rank_spaces + self.rank + "|")
        card_rows.append("|"+ card_width*" " +"|")
        card_rows.append("|"+ int(card_width/2) * " " + suits[self.suit] + suit_spaces* " "+ "|")
        card_rows.append("|"+ card_width*" " +"|")
        card_rows.append("|"+  self.rank  + rank_spaces +"|")
        card_rows.append("-" *(card_width+2))
        return card_rows

    def pretty_print(self):
        for row in self.pretty_string():
            print(row)

class Deck:
    def __init__(self):
        self.deck = []
        for suit in SUITS:
            for rank in RANKS:
                self.deck.append(Card(rank, suit))

    def shuffle(self):
        random.shuffle(self.deck)

    def deal(self):
        return self.deck.pop()

class Hand:
    def __init__(self):
        self.cards = []

    def add_card(self, card):
        self.cards.append(card)

    def __str__(self):
        return f"Cards: {', '.join(str(c) for c in self.cards)}"

    def pretty_print(self):
        self.sort_hand()
        card_height=7
        pretty_cards=[]
        for card in self.cards:
            pretty_cards.append(card.pretty_string())
        
        for row in range(card_height):
            for card in pretty_cards:
                print(card[row],end=" ")
            print("")

    def sort_hand(self):
        self.cards= sorted(self.cards, key=lambda x:x.value, reverse=True)

    
    def check_hand(self):
        if not self.cards:
            return ("No Cards",'0','0')
        
        self.sort_hand()
        suits = dict([(s,[]) for s in SUITS])
        duplicates={}
        for card in self.cards:
            #group cards by suit
            suits[card.suit].append(card.value)
            #look for pairs 4 of a kind etc
            duplicates[card.value] = duplicates.get(card.value,0)+1
        

        #check for straight flush
        high_card=0
        for suit in suits:
            high_card = max(self._is_straight(suits[suit]),high_card)
            if high_card:
                return ("Straight Flush", high_card, high_card)
        
        
        #check for duplicate based hands
        #sort by highest rank to lowest rank
        ranks = sorted(duplicates.keys(), reverse=True)
        high_cards = ranks[:3]
        three_of_a_kind = False
        pairs = []
        for k in ranks:
            if duplicates[k] == 4:
                #if there is a four of a kind grab the highest card, 
                #if highest is the four of a kind grab second
                high_card=k
                if len(high_cards)>1:
                    high_card = high_cards[1] if k == high_cards[0] else high_cards[0]
                return ("Four of a Kind",k,high_card)
            
            if duplicates[k]==3 and not three_of_a_kind:
                three_of_a_kind = k

            if duplicates[k]==2:
                pairs.append(k)
        
        if three_of_a_kind and pairs:
            return ("Full House",three_of_a_kind, pairs[0])
        
        #check for flush and straight
        #flush
        high_card=0
        for suit in suits:
            if len(suits[suit])>=5:
                high_card = max(suits[suit][0], high_card)
        if high_card:
            return("Flush",high_card, high_card)
        
        #straight
        high_card=self._is_straight(ranks)
        if high_card:
            return("Straight",high_card ,high_card)
        

        #back to pairs
        if len(pairs)>=2:
            return("Two Pair",pairs[0],pairs[1])
            #its possible two players could have the same two pair match,
            #additional high card required
        elif pairs:
            high_card=pairs[0]
            if len(high_cards)>1:
                high_card= high_cards[1] if pairs[0] == high_cards[0] else high_cards[0]
            return ("Pair",pairs[0],high_card)
        
        return ("High Card", high_cards[0],high_cards[0])

        
    def _is_straight(self, desc_list):
        left, right = 0, 1
        while right < len(desc_list):
            sequence_len = right - left +1
            edge_case = sequence_len >= 4 and desc_list[right] == 2
            if sequence_len >= 5 or edge_case:
                return desc_list[left]
            
            #if not sequential reset the left window
            if desc_list[right-1] != desc_list[right]+1:
                left = right+1
            right+=1
            
        return False    

def play_basic_game():
    # Create a deck, shuffle it, and deal 5 cards to the player
    deck = Deck()
    deck.shuffle()
    player_hand = Hand()
    for _ in range(7):
        player_hand.add_card(deck.deal())
    hand, high_rank,high_card=player_hand.check_hand()
    reverse_value={14:"A",13:"K",12:"Q",11:"J"}
    high_rank= reverse_value.get(high_rank,high_rank)
    high_card= reverse_value.get(high_card,high_card)
    print(f"Hand: {hand}\nHighest Rank: {high_rank}\n2nd Rank / High Card: {high_card}")

    # Print the player's hand using pretty_print
    player_hand.pretty_print()



if __name__ == "__main__":
    play_basic_game()
