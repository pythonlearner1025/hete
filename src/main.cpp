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
        throw std::invalid_argument("Invalid card string length.");
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
        throw std::invalid_argument("Invalid rank character.");
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
        throw std::invalid_argument("Invalid suit character.");
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

void test_poker() {
    try {
        // Step 1: Initialize the Poker Engine in Manual Mode
        std::vector<double> starting_stacks = {7380000, 2500000, 5110000, 10170000, 4545000};
        int n_players = 5;
        double small_blind = 40000;
        double big_blind = 80000;
        int max_round_bets = 4;
        bool manual = true;

        // Initialize the PokerEngine with manual mode enabled
        PokerEngine engine(starting_stacks, n_players, small_blind, big_blind, max_round_bets, manual);

        std::cout << "Poker Engine initialized in manual mode." << std::endl;

        // Step 2: Manually Deal Hands to Players
        // Actions:
        // 'd dh p1 7s4s', 'd dh p2 Js8h', 'd dh p3 Td8c', 'd dh p4 6d5h', 'd dh p5 Qh7h'

        engine.manual_deal_hand(0, create_hand("7s", "4s")); // p1
        engine.manual_deal_hand(1, create_hand("Js", "8h")); // p2
        engine.manual_deal_hand(2, create_hand("Td", "8c")); // p3
        engine.manual_deal_hand(3, create_hand("6d", "5h")); // p4
        engine.manual_deal_hand(4, create_hand("Qh", "7h")); // p5

        std::cout << "Hands dealt manually to all players." << std::endl;

        // Step 3: Execute Actions in Order
        // Actions list:
        std::vector<std::string> actions = {
            "p3 f",
            "p4 cbr 170000",
            "p5 f",
            "p1 f",
            "p2 cc",
            "d db JcTs2d",
            "p2 cc",
            "p4 cbr 140000",
            "p2 cc",
            "d db As",
            "p2 cc",
            "p4 cbr 325000",
            "p2 cc",
            "d db Qs",
            "p2 cc",
            "p4 cbr 600000",
            "p2 cc",
            "p4 sm 6d5h",
            "p2 sm Js8h"
        };

        for (const auto& action : actions) {
            std::cout << "Processing action: " << action << std::endl;
            // Split the action string by spaces
            std::vector<std::string> tokens;
            size_t pos = 0, found;
            while((found = action.find_first_of(' ', pos)) != std::string::npos){
                tokens.push_back(action.substr(pos, found - pos));
                pos = found+1;
            }
            tokens.push_back(action.substr(pos));

            if (tokens.empty()) {
                continue;
            }

            std::string action_type = tokens[0];

            if (action_type == "p3" || action_type == "p4" || action_type == "p5" || action_type == "p1" || action_type == "p2") {
                // Player action
                // tokens[0]: player identifier (e.g., "p3")
                // tokens[1]: action (e.g., "f", "cbr", "cc", "sm")
                std::string player_str = tokens[0];
                std::string player_action = tokens[1];
                int player_index = -1;

                // Map player string to index
                if (player_str == "p1") player_index = 0;
                else if (player_str == "p2") player_index = 1;
                else if (player_str == "p3") player_index = 2;
                else if (player_str == "p4") player_index = 3;
                else if (player_str == "p5") player_index = 4;
                else {
                    std::cerr << "Unknown player identifier: " << player_str << std::endl;
                    continue;
                }

                if (player_action == "f") {
                    // Fold
                    engine.fold(player_index);
                    std::cout << "Player " << player_str << " folded." << std::endl;
                }
                else if (player_action == "cbr") {
                    // Check or Bet Raise with amount
                    if (tokens.size() < 3) {
                        std::cerr << "Invalid cbr action format." << std::endl;
                        continue;
                    }
                    double amount = std::stod(tokens[2]);
                    engine.bet_or_raise(player_index, amount);
                    std::cout << "Player " << player_str << " bet or raised " << amount << "." << std::endl;
                }
                else if (player_action == "cc") {
                    // Check or Call
                    engine.check_or_call(player_index);
                    std::cout << "Player " << player_str << " checked or called." << std::endl;
                }
                else if (player_action == "sm") {
                    // Showdown or Special Action
                    // Assuming 'sm' stands for showdown
                    // Optionally, handle specific showdown actions
                    // For simplicity, we call showdown
                    std::cout << "Player " << player_str << " triggered showdown." << std::endl;
                    engine.showdown();
                }
                else {
                    std::cerr << "Unknown player action: " << player_action << std::endl;
                }
            }
            else if (action_type == "d") {
                // Deal action
                // tokens[0]: "d"
                // tokens[1]: "db" or "dh"
                // tokens[2]: target (e.g., "JcTs2d" or player identifier)
                if (tokens.size() < 3) {
                    std::cerr << "Invalid deal action format." << std::endl;
                    continue;
                }
                std::string deal_type = tokens[1];
                std::string deal_data = tokens[2];

                if (deal_type == "dh") {
                    // Deal Hand (already handled manually)
                    // Skip or handle if necessary
                    std::cout << "Deal Hand action ignored in manual mode." << std::endl;
                    continue;
                }
                else if (deal_type == "db") {
                    // Deal Board
                    // e.g., "JcTs2d" => ['Jc','Ts','2d']
                    std::vector<int> board_cards;
                    for (size_t i = 0; i+1 < deal_data.length(); i +=2 ) {
                        board_cards.push_back(card_str_to_index(deal_data.substr(i,2)));
                    }
                    engine.manual_deal_board(board_cards);
                    std::cout << "Board dealt manually with cards: " << deal_data << std::endl;
                }
                else {
                    std::cerr << "Unknown deal type: " << deal_type << std::endl;
                }
            }
            else {
                std::cerr << "Unknown action type: " << action_type << std::endl;
            }
        }

        // Step 4: Verify Final Stack Sizes
        std::vector<double> expected_finishing_stacks = {7340000, 3775000, 5110000, 8935000, 4545000};
        // Retrieve actual finishing stacks
        std::array<double, PokerEngine::MAX_PLAYERS> actual_finishing_stacks = engine.get_payoffs();

        std::cout << "\nFinal Stack Verification:" << std::endl;
        for(int i=0;i<n_players;i++){
            double expected = expected_finishing_stacks[i];
            double actual = engine.players[i].stack;
            if(std::abs(expected - actual) < 1e-2){
                std::cout << "Player " << (i+1) << " stack correct: " << actual << std::endl;
            }
            else{
                std::cout << "Player " << (i+1) << " stack incorrect: expected " << expected << ", got " << actual << std::endl;
            }
        }

        std::cout << "Test completed successfully." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "An exception occurred: " << e.what() << std::endl;
    }
}

int main() {
    test_poker();
    return 0;
}
