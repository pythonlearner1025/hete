// main.cpp
#include "engine.h"
#include "ParsedPHH.h"
#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <cctype>
#include <filesystem>
#include "ParsedPHH.h"
#include <fstream>
#include <sstream>
#include <cassert>

namespace fs = std::filesystem;
// Helper function to trim whitespace from both ends of a string
std::string trim(const std::string& s) {
    size_t start = s.find_first_not_of(" \t\r\n");
    size_t end = s.find_last_not_of(" \t\r\n");
    return (start == std::string::npos) ? "" : s.substr(start, end - start + 1);
}

// Function to parse a list from a string, e.g., [1, 2, 3]
std::vector<double> parse_list(const std::string& list_str) {
    std::vector<double> list;
    size_t start = list_str.find('[');
    size_t end = list_str.find(']');
    if (start == std::string::npos || end == std::string::npos || end < start) {
        throw std::invalid_argument("Invalid list format: " + list_str);
    }
    std::string elements = list_str.substr(start + 1, end - start - 1);
    std::stringstream ss(elements);
    std::string item;
    while (std::getline(ss, item, ',')) {
        list.push_back(std::stod(trim(item)));
    }
    return list;
}

// Function to parse actions from a string, e.g., ['action1', 'action2']
std::vector<std::string> parse_actions(const std::string& actions_str) {
    std::vector<std::string> actions;
    size_t start = actions_str.find('[');
    size_t end = actions_str.find(']');
    if (start == std::string::npos || end == std::string::npos || end < start) {
        throw std::invalid_argument("Invalid actions format: " + actions_str);
    }
    std::string elements = actions_str.substr(start + 1, end - start - 1);
    std::stringstream ss(elements);
    std::string item;
    while (std::getline(ss, item, '\'')) { // Actions are enclosed in single quotes
        // Skip commas and spaces
        item = trim(item);
        if (item.empty() || item == ",") continue;
        actions.push_back(item);
    }
    return actions;
}

// Function to parse a .phh file and populate ParsedPHH structure
bool parse_phh(const std::string& file_path, ParsedPHH& phh) {
    std::ifstream infile(file_path);
    if (!infile.is_open()) {
        std::cerr << "Failed to open file: " << file_path << std::endl;
        return false;
    }

    std::string line;
    bool has_blinds_or_straddles = false;

    // Temporary storage for parsed data
    std::vector<double> starting_stacks;
    std::vector<double> finishing_stacks;
    std::vector<double> antes;
    double min_bet = 0.0;
    std::vector<std::string> actions;
    bool is_NT = false;

    while (std::getline(infile, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') continue; // Skip empty lines and comments

        if (line.find("variant = 'NT'") != std::string::npos) {
            is_NT = true;
        }

        // Parse blinds_or_straddles
        if (line.find("blinds_or_straddles") != std::string::npos) {
            has_blinds_or_straddles = true;
            size_t eq_pos = line.find('=');
            if (eq_pos != std::string::npos) {
                std::string list_str = line.substr(eq_pos + 1);
                std::vector<double> blinds_vec = parse_list(list_str);
                if (blinds_vec.size() >= 2) {
                    phh.small_bet = blinds_vec[0];
                    phh.big_bet = blinds_vec[1];
                }
            }
        }

        // Parse starting_stacks
        if (line.find("starting_stacks") != std::string::npos) {
            size_t eq_pos = line.find('=');
            if (eq_pos == std::string::npos) continue;
            std::string list_str = line.substr(eq_pos + 1);
            starting_stacks = parse_list(list_str);
            continue;
        }

        // Parse finishing_stacks
        if (line.find("finishing_stacks") != std::string::npos) {
            size_t eq_pos = line.find('=');
            if (eq_pos == std::string::npos) continue;
            std::string list_str = line.substr(eq_pos + 1);
            finishing_stacks = parse_list(list_str);
            continue;
        }

        // Parse antes
        if (line.find("antes") != std::string::npos) {
            size_t eq_pos = line.find('=');
            if (eq_pos == std::string::npos) continue;
            std::string list_str = line.substr(eq_pos + 1);
            antes = parse_list(list_str);
            continue;
        }

        // Parse min_bet
        if (line.find("min_bet") != std::string::npos) {
            size_t eq_pos = line.find('=');
            if (eq_pos == std::string::npos) continue;
            std::string bet_str = line.substr(eq_pos + 1);
            std::cout << "min_bet = " + bet_str;
            min_bet = std::stod(trim(bet_str));
            continue;
        } 

        // Parse actions
        if (line.find("actions") != std::string::npos) {
            size_t eq_pos = line.find('=');
            if (eq_pos == std::string::npos) continue;
            std::string actions_str = line.substr(eq_pos + 1);
            actions = parse_actions(actions_str);
            continue;
        }
    }

    infile.close();

    if (!is_NT) {
        std::cerr << "Phh isn't NT: " << file_path << std::endl;
        return false;
    }

    // Validate that all necessary fields were parsed
    if (starting_stacks.empty()) {
        std::cerr << "Missing starting_stacks in file: " << file_path << std::endl;
        return false;
    }
    if (finishing_stacks.empty()) {
        std::cerr << "Missing finishing_stacks in file: " << file_path << std::endl;
        return false;
    }
    if (antes.empty()) {
        std::cerr << "Missing antes in file: " << file_path << std::endl;
        return false;
    }
    if (actions.empty()) {
        std::cerr << "Missing actions in file: " << file_path << std::endl;
        return false;
    }

    std::array<double, PHH_NUM_PLAYERS> start_stacks_array;
    std::array<double, PHH_NUM_PLAYERS> finish_stacks_array;
    std::array<double, PHH_NUM_PLAYERS> antes_array;

    for (size_t i=0; i<PHH_NUM_PLAYERS; ++i) {
        start_stacks_array[i] = starting_stacks[i];
        finish_stacks_array[i] = finishing_stacks[i];
        antes_array[i] = antes[i];
    }
    phh.starting_stacks = start_stacks_array;
    phh.finishing_stacks = finish_stacks_array;

    phh.n_players = static_cast<int>(starting_stacks.size());
    if (!has_blinds_or_straddles){
        phh.small_bet = min_bet;
        phh.big_bet = min_bet * 2; // Assuming big_bet is twice the small_bet
    }
    phh.antes = antes_array;
    phh.actions = actions;
    std::cout << "\n-- Processing phh " + file_path + " --\n";
    return true;
}

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
        std::cout << "Player " << player_str << " folded." << std::endl;
        engine.fold(player_index);
    }
    else if (action == "cbr") {
        // Check, Bet, or Raise with amount
        if (params.size() < 1) {
            throw std::invalid_argument("Missing amount for 'cbr' action.");
        }
        double amount = std::stod(params[0]);
        std::cout << "Player " << player_str << " bet/raised " << amount << "." << std::endl;
        engine.bet_or_raise(player_index, amount);
    }
    else if (action == "cc") {
        // Check or Call
        std::cout << "Player " << player_str << " checked/called." << std::endl;
        engine.check_or_call(player_index);
    }
    else if (action == "sm") {
        // Showdown
        std::cout << "Showdown was triggered by " << player_str << "." << std::endl;
        engine.showdown();
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
        std::array<int, 2> cards;
        for (size_t i = 0; i + 1 < cards_str.length(); i += 2) {
            cards[i] = card_str_to_index(cards_str.substr(i, 2));
        }

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
        engine.manual_deal_hand(player_index, cards);
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

        std::array<int, 5> board;
        for (size_t i = 0; i + 1 < cards_str.length(); i += 2) {
            board[i] = card_str_to_index(cards_str.substr(i, 2));
        }

        engine.manual_deal_board(board);
        std::cout << "Board dealt manually with cards: " << cards_str << std::endl;
    }
    else {
        throw std::invalid_argument("Unknown deal type: " + deal_type);
    }
}

// Modified test_poker function
bool test_poker(const ParsedPHH& phh, const std::string& file_name) {
    try {
        // Initialize the PokerEngine with parsed parameters
        // Assuming constructor: PokerEngine(starting_stacks, n_players, small_bet, big_bet, antes, actions, manual_mode)
        // Adjust constructor parameters as per your implementation

        // Extract parameters
        std::array<double, PHH_NUM_PLAYERS> starting_stacks = phh.starting_stacks;
        int n_players = phh.n_players;
        double small_bet = phh.small_bet;
        double big_bet = phh.big_bet;
        std::array<double, PHH_NUM_PLAYERS> antes = phh.antes;

        std::vector<std::string> actions = phh.actions;
        int starting_actor = 2;

        if (!actions.empty() && actions[actions.size() - 1][0] == 'd') {
            throw std::runtime_error("showdown before preflop is not handled");
        }

        // Print init values
        std::cout << "Starting stacks: ";
        for (const auto& stack : starting_stacks) {
            std::cout << stack << " ";
        }
        std::cout << "\nNumber of players: " << n_players;
        std::cout << "\nSmall bet: " << small_bet;
        std::cout << "\nBig bet: " << big_bet;
        std::cout << "\nAntes: ";
        for (const auto& ante : antes) {
            std::cout << ante << " ";
        }
        std::cout << "\nStarting actor: " << starting_actor << std::endl;

        // Initialize PokerEngine
        PokerEngine engine(
            starting_stacks, 
            antes,
            starting_actor,
            n_players, 
            small_bet, 
            big_bet, 
            true
        ); // Assuming max_round_bets = 4 and manual_mode = true

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

        // After processing actions, verify finishing stacks
        std::array<double, MAX_PLAYERS> actual_finishing_stacks = engine.get_finishing_stacks();

        // Compare actual_finishing_stacks with phh.finishing_stacks
        for (int i = 0; i < n_players; ++i) {
            double expected = phh.finishing_stacks[i];
            double actual = actual_finishing_stacks[i];
            if (std::abs(expected - actual) > 1e-2) {
                std::cerr << "Mismatch in player " << (i + 1) << " stack. Expected: " << expected << ", Actual: " << actual << std::endl;
                return false;
            }
        }

        // Print out construct_history values
        auto [bet_status, bet_fracs] = engine.construct_history();
        std::cout << "Construct History Results:" << std::endl;
        std::cout << "Bet Status: ";
        for (bool status : bet_status) {
            std::cout << status << " ";
        }
        std::cout << std::endl;
        std::cout << "Bet Fractions: ";
        for (double frac : bet_fracs) {
            std::cout << frac << " ";
        }
        std::cout << std::endl;

        return true;
    } catch (const std::exception& e) {
        std::cerr << "Exception in test_poker for file " << file_name << ": " << e.what() << std::endl;
        return false;
    }
}

int run() {
    // Define the directory containing .phh files
    const std::string directory_path = "/Users/minjunes/poker/phh-dataset/data/wsop/2023/43/5"; // Change this to your directory path

    // Initialize statistics
    int total_tests = 0;
    int passed_tests = 0;
    std::vector<std::string> failed_files;

    // Iterate through all .phh files in the directory
    for (const auto& entry : fs::directory_iterator(directory_path)) {
        if (entry.is_regular_file() && entry.path().extension() == ".phh") {
            std::string file_path = entry.path().string();
            std::string file_name = entry.path().filename().string();

            ParsedPHH phh;

            // Parse the .phh file
            bool parsed = parse_phh(file_path, phh);
            if (!parsed) {
                std::cout << "Skipping file (contains 'blinds_or_straddles'): " << file_name << std::endl;
                continue;
            }

            // Run the test_poker function
            bool test_passed = test_poker(phh, file_name);

            // Update statistics
            total_tests++;
            if (test_passed) {
                passed_tests++;
                std::cout << "Test Passed for file: " << file_name << std::endl;
            }
            else {
                failed_files.push_back(file_name);
                std::cout << "Test Failed for file: " << file_name << std::endl;
            }
        }
    }

    // Calculate pass percentage
    double pass_percentage = (total_tests > 0) ? ((static_cast<double>(passed_tests) / total_tests) * 100.0) : 0.0;

    // Print statistics
    std::cout << "\n===== Test Summary =====" << std::endl;
    std::cout << "Total .phh Tests Simulated: " << total_tests << std::endl;
    std::cout << "Passed: " << passed_tests << " (" << pass_percentage << "%)" << std::endl;
    std::cout << "Failed: " << failed_files.size() << " (" << ((total_tests > 0) ? (100.0 - pass_percentage) : 0.0) << "%)" << std::endl;

    if (!failed_files.empty()) {
        std::cout << "\n===== Failed .phh Files =====" << std::endl;
        for (const auto& fname : failed_files) {
            std::cout << fname << std::endl;
        }
    }

    return 0;
}
