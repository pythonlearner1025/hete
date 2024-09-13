// test_poker.cpp

#include "engine.h"
#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <cctype>

// Helper Function to Convert Card String to Index
int card_str_to_index(const std::string& card_str) {
    if (card_str.length() != 2) {
        throw std::invalid_argument("Invalid card string length: " + card_str);
    }

    // Map Rank
    char rank_char = std::toupper(card_str[0]);
    int rank_index;
    if (rank_char >= '2' && rank_char <= '9') {
        rank_index = rank_char - '2';
    }
    else if (rank_char == 'T') {
        rank_index = 8;
    }
    else if (rank_char == 'J') {
        rank_index = 9;
    }
    else if (rank_char == 'Q') {
        rank_index = 10;
    }
    else if (rank_char == 'K') {
        rank_index = 11;
    }
    else if (rank_char == 'A') {
        rank_index = 12;
    }
    else {
        throw std::invalid_argument("Invalid rank character in card: " + card_str);
    }

    // Map Suit
    char suit_char = std::tolower(card_str[1]);
    int suit_index;
    if (suit_char == 's') {
        suit_index = 0;
    }
    else if (suit_char == 'h') {
        suit_index = 1;
    }
    else if (suit_char == 'd') {
        suit_index = 2;
    }
    else if (suit_char == 'c') {
        suit_index = 3;
    }
    else {
        throw std::invalid_argument("Invalid suit character in card: " + card_str);
    }

    return 4 * rank_index + suit_index;
}

// Helper Function to Create Hand from Two Card Strings
omp::Hand create_hand(const std::string& card1, const std::string& card2) {
    int card1_idx = card_str_to_index(card1);
    int card2_idx = card_str_to_index(card2);
    return omp::Hand(card1_idx) + omp::Hand(card2_idx);
}

// Helper Function to Create Hand from Multiple Card Strings
omp::Hand create_hand(const std::vector<std::string>& cards) {
    if (cards.empty()) {
        return omp::Hand::empty();
    }

    omp::Hand hand(card_str_to_index(cards[0]));
    for (size_t i = 1; i < cards.size(); ++i) {
        hand += omp::Hand(card_str_to_index(cards[i]));
    }
    return hand;
}

// Function to Process Player Actions
void process_player_action(PokerEngine& engine, const std::string& player_str, const std::string& action, const std::vector<std::string>& params = {}) {
    int player_index = -1;

    // Map player string to index
    if (player_str == "p1") player_index = 0;
    else if (player_str == "p2") player_index = 1;
    else if (player_str == "p3") player_index = 2;
    else if (player_str == "p4") player_index = 3;
    else if (player_str == "p5") player_index = 4;
    else {
        throw std::invalid_argument("Unknown player identifier: " + player_str);
    }

    if (action == "f") {
        // Fold
        engine.fold(player_index);
        std::cout << "Player " << player_str << " folded." << std::endl;
    }
    else if (action == "cbr") {
        // Check, Bet, or Raise with amount
        if (params.size() < 1) {
            throw std::invalid_argument("Missing amount for 'cbr' action.");
        }
        double amount = std::stod(params[0]);
        engine.bet_or_raise(player_index, amount);
        std::cout << "Player " << player_str << " bet/raised " << amount << "." << std::endl;
    }
    else if (action == "cc") {
        // Check or Call
        engine.check_or_call(player_index);
        std::cout << "Player " << player_str << " checked/called." << std::endl;
    }
    else if (action == "sm") {
        // Showdown
        engine.showdown();
        std::cout << "Showdown triggered by " << player_str << "." << std::endl;
    }
    else {
        throw std::invalid_argument("Unknown player action: " + action);
    }
}

// Function to Process Deal Actions
void process_deal_action(PokerEngine& engine, const std::vector<std::string>& tokens) {
    if (tokens.size() < 3) {
        throw std::invalid_argument("Invalid deal action format.");
    }

    std::string deal_type = tokens[1];

    if (deal_type == "dh") {
        // Deal Hand
        if (tokens.size() < 4) {
            throw std::invalid_argument("Invalid 'd dh' action format. Expected: 'd dh pX card1card2'");
        }
        std::string player_str = tokens[2];
        std::string cards_str = tokens[3];

        // Extract individual card strings (each card is 2 characters)
        if (cards_str.length() != 4) {
            throw std::invalid_argument("Invalid number of cards for 'dh' action. Expected 2 cards.");
        }
        std::vector<std::string> card_strings;
        for (size_t i = 0; i + 1 < cards_str.length(); i += 2) {
            card_strings.push_back(cards_str.substr(i, 2));
        }
        if (card_strings.size() != 2) {
            throw std::invalid_argument("Invalid number of cards for 'dh' action. Expected 2 cards.");
        }

        omp::Hand hand = create_hand(card_strings);

        // Map player string to index
        int player_index = -1;
        if (player_str == "p1") player_index = 0;
        else if (player_str == "p2") player_index = 1;
        else if (player_str == "p3") player_index = 2;
        else if (player_str == "p4") player_index = 3;
        else if (player_str == "p5") player_index = 4;
        else {
            throw std::invalid_argument("Unknown player identifier in 'dh' action: " + player_str);
        }

        // Deal the hand to the player
        engine.manual_deal_hand(player_index, hand);
        std::cout << "Dealt hand " << cards_str << " to " << player_str << "." << std::endl;
    }
    else if (deal_type == "db") {
        // Deal Board
        if (tokens.size() < 3) {
            throw std::invalid_argument("Invalid 'd db' action format. Expected: 'd db card1card2card3...'");
        }
        std::string cards_str = tokens[2];

        // Each card is 2 characters
        if (cards_str.length() % 2 != 0) {
            throw std::invalid_argument("Invalid board cards string length.");
        }

        std::vector<int> board;
        for (size_t i = 0; i + 1 < cards_str.length(); i += 2) {
            board.push_back(card_str_to_index(cards_str.substr(i, 2)));
        }

        engine.manual_deal_board(board);
        std::cout << "Board dealt manually with cards: " << cards_str << std::endl;
    }
    else {
        throw std::invalid_argument("Unknown deal type: " + deal_type);
    }
}

void test_poker() {
    try {
        // Step 1: Initialize the Poker Engine in Manual Mode
        std::vector<double> starting_stacks = {2400000, 9700000, 4575000, 8175000, 4850000};
        std::vector<double> expected_finishing_stacks = {2600000, 9550000, 4475000, 8075000, 4750000};
        int n_players = 5;
        double small_bet = 200000;
        double big_bet = 400000;
        double bring_in = 50000;
        int max_round_bets = 4;
        bool manual = true;
        int actor = 2;

        // Initialize the PokerEngine with manual mode enabled
        PokerEngine engine(starting_stacks, actor, n_players, small_bet, big_bet, max_round_bets, manual);

        std::cout << "Poker Engine initialized in manual mode." << std::endl;

        // Step 2: Execute Actions in Order
        // Actions list as per the .phh file:
        std::vector<std::string> actions = {
            "d dh p1 Qh6cJd",
            "d dh p2 8s5d3c",
            "d dh p3 7d4dTs",
            "d dh p4 6h3dTd",
            "d dh p5 Qc6dTh",
            "p2 pb",
            "p3 f",
            "p4 f",
            "p5 f",
            "p1 cbr 200000",
            "p2 f"
        };

        for (const auto& action_str : actions) {
            std::cout << "\nProcessing action: " << action_str << std::endl;
            // Split the action string by spaces
            std::vector<std::string> tokens;
            size_t pos = 0, found;
            while ((found = action_str.find_first_of(' ', pos)) != std::string::npos) {
                tokens.push_back(action_str.substr(pos, found - pos));
                pos = found + 1;
            }
            tokens.push_back(action_str.substr(pos));

            if (tokens.empty()) {
                continue;
            }

            std::string first_token = tokens[0];

            if (first_token == "d") {
                // Deal action
                try {
                    process_deal_action(engine, tokens);
                }
                catch (const std::exception& e) {
                    std::cerr << "Error processing deal action: " << e.what() << std::endl;
                }
            }
            else {
                // Player action
                try {
                    if (tokens.size() < 2) {
                        throw std::invalid_argument("Invalid player action format: " + action_str);
                    }
                    std::string player_str = tokens[0];
                    std::string action = tokens[1];
                    std::vector<std::string> params;
                    for (size_t i = 2; i < tokens.size(); ++i) {
                        params.push_back(tokens[i]);
                    }
                    process_player_action(engine, player_str, action, params);
                }
                catch (const std::exception& e) {
                    std::cerr << "Error processing player action: " << e.what() << std::endl;
                }
            }
        }

        // Step 3: Verify Final Stack Sizes
        std::array<double, PokerEngine::MAX_PLAYERS> actual_finishing_stacks = engine.get_finishing_stacks();

        std::cout << "\nFinal Stack Verification:" << std::endl;
        bool all_correct = true;
        for(int i = 0; i < n_players; i++) {
            double expected = expected_finishing_stacks[i];
            double actual = actual_finishing_stacks[i];
            if(std::abs(expected - actual) < 1e-2){
                std::cout << "Player " << (i+1) << " stack correct: " << actual << std::endl;
            }
            else{
                std::cout << "Player " << (i+1) << " stack incorrect: expected " << expected << ", got " << actual << std::endl;
                all_correct = false;
            }
        }

        if(all_correct){
            std::cout << "\nTest completed successfully. All player stacks are correct." << std::endl;
        }
        else{
            std::cout << "\nTest failed. Some player stacks are incorrect." << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error occurred during test: " << e.what() << std::endl;
    }
}

int main() {
    test_poker();
    return 0;
}
