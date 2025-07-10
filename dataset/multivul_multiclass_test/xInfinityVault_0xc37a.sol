// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

/**
 * @title xInfinityVault
 * @dev A decentralized investment protocol with the following features:
 * - 1% daily ROI on all deposits
 * - 5-level referral system with bonuses (5%, 3%, 2%, 1%, 0.5%)
 * - 3% marketing fee on all transactions (deposits and withdrawals)
 * - Minimum deposit: 0.001 ETH
 * - Maximum 100 deposits per address
 *
 * Security Features:
 * - No external dependencies
 * - Simple and auditable code
 * - Automatic ROI calculations
 * - Fair and transparent reward distribution
 *
 * How it works:
 * 1. Users deposit ETH (min 0.001)
 * 2. Daily ROI of 1% is calculated based on deposit time
 * 3. Users can withdraw earnings at any time
 * 4. Referral rewards are paid instantly on deposits
 * 5. Marketing fee (3%) is collected on all transactions
 */

interface IERC20 {
    /**
     * @dev Returns the amount of tokens owned by `account`.
     */
    function balanceOf(address account) external view returns (uint256);
}


abstract contract ReentrancyGuard {
    bool internal locked;

    modifier noReentrant() {
        require(!locked, "No re-entrancy");
        locked = true;
        _;
        locked = false;
    }
}

contract xInfinityVault is ReentrancyGuard {
    address public owner;
    address public constant MARKETING_WALLET = 0x8a7198c128587E89a8d21cfFe5F1e3ED38FA7be5; 
    

    uint256 public invested;
    uint256 public withdrawn;
    uint256 public match_bonus;
    uint256 public marketing_fees;
    address[] public stakers; 

    
    uint256 constant BONUS_LINES_COUNT = 5;
    uint256 constant PERCENT_DIVIDER = 1000; 
    uint256[BONUS_LINES_COUNT] public ref_bonuses = [50, 30, 20, 10, 5]; 

    struct Deposit {
        uint256 amount;
        uint256 time;
    }

    struct Player {
        address upline;
        uint256 dividends;
        uint256 match_bonus;
        uint256 last_payout;
        uint256 total_invested;
        uint256 total_withdrawn;
        uint256 total_match_bonus;
        Deposit[] deposits;
        uint256[5] structure;
    }

    mapping(address => Player) public players;

    event Upline(address indexed addr, address indexed upline, uint256 bonus);
    event NewDeposit(address indexed addr, uint256 amount);
    event MatchPayout(address indexed addr, address indexed from, uint256 amount);
    event Withdraw(address indexed addr, uint256 amount);

    constructor() {
        owner = msg.sender;
    }

    /**
     * @dev Internal function to calculate and update user payouts
     * Processes the ROI earnings for all user deposits
     * Updates the last payout timestamp and adds earnings to dividends
     */
    function _payout(address _addr) private {
        uint256 payout = this.payoutOf(_addr);

        if(payout > 0) {
            players[_addr].last_payout = block.timestamp;
            players[_addr].dividends += payout;
        }
    }

    /**
     * @dev Internal function to handle referral payments
     * @param _addr Address of the depositor
     * @param _amount Amount to calculate referral bonuses from
     * Distributes referral bonuses across 5 levels:
     * - Level 1: 5.0% (50/1000)
     * - Level 2: 3.0% (30/1000)
     * - Level 3: 2.0% (20/1000)
     * - Level 4: 1.0% (10/1000)
     * - Level 5: 0.5% (5/1000)
     */
    function _refPayout(address _addr, uint256 _amount) private {
        address up = players[_addr].upline;

        for(uint8 i = 0; i < ref_bonuses.length; i++) {
            if(up == address(0)) break;
            
            uint256 bonus = _amount * ref_bonuses[i] / PERCENT_DIVIDER;
            
            players[up].match_bonus += bonus;
            players[up].total_match_bonus += bonus;

            match_bonus += bonus;

            emit MatchPayout(up, _addr, bonus);

            up = players[up].upline;
        }
    }

    /**
     * @dev Internal function to set up referral relationships
     * @param _addr Address of the new user
     * @param _upline Address of the referrer
     * @param _amount Deposit amount (used for bonus calculation)
     *
     * Rules:
     * - Only sets upline for new users
     * - If upline has no deposits, defaults to owner
     * - Updates referral structure up to 5 levels deep
     * - Emits Upline event with 1% bonus amount
     */
    function _setUpline(address _addr, address _upline, uint256 _amount) private {
        if(players[_addr].upline == address(0) && _addr != owner) {
            if(players[_upline].deposits.length == 0) {
                _upline = owner;
            }

            players[_addr].upline = _upline;

            emit Upline(_addr, _upline, _amount / 100);
            
            for(uint8 i = 0; i < BONUS_LINES_COUNT; i++) {
                players[_upline].structure[i]++;

                _upline = players[_upline].upline;

                if(_upline == address(0)) break;
            }
        }
    }
    
    /**
     * @dev Allows users to make a deposit and enter the protocol
     * @param _upline Referrer's address
     * Requirements:
     * - Minimum deposit: 0.001 ETH
     * - Maximum 100 deposits per address
     * Fees:
     * - 1% marketing fee sent to MARKETING_WALLET
     * - Remaining 99% added to user's deposit balance
     * Effects:
     * - Updates user's deposit history
     * - Triggers referral payments if applicable
     * - Emits NewDeposit event
     */
    function deposit(address _upline) external payable noReentrant {
        require(msg.value >= 0.001 ether, "Minimum deposit amount is 0.001 ETH");

        Player storage player = players[msg.sender];

        if (player.deposits.length == 0) {
            stakers.push(msg.sender);
        }

        require(player.deposits.length < 100, "Max 100 deposits per address");

        _setUpline(msg.sender, _upline, msg.value);

        // Calculate fees
        uint256 marketingFee = msg.value / 300; // 3% marketing fee
        uint256 depositAmount = msg.value - marketingFee;

        player.deposits.push(Deposit({
            amount: depositAmount,
            time: block.timestamp
        }));

        player.total_invested += depositAmount;
        invested += depositAmount;
        marketing_fees += marketingFee;

        _refPayout(msg.sender, depositAmount);

        payable(MARKETING_WALLET).transfer(marketingFee);
        
        emit NewDeposit(msg.sender, depositAmount);
    }
    
    /**
     * @dev Allows users to withdraw their earnings
     * Includes:
     * - ROI earnings from deposits
     * - Referral bonuses
     * Fees:
     * - 1% marketing fee sent to MARKETING_WALLET
     * - Remaining 99% sent to user
     * Effects:
     * - Updates user's withdrawal history
     * - Resets dividends and match bonus to zero
     * - Emits Withdraw event
     */
    function withdraw() external noReentrant{
        Player storage player = players[msg.sender];
        require(player.total_invested > 0, "Join first");

        _payout(msg.sender);

        require(player.dividends > 0 || player.match_bonus > 0, "Zero amount");

        uint256 amount = player.dividends + player.match_bonus;
        uint256 marketingFee = amount / 300; // 3% marketing fee
        uint256 withdrawAmount = amount - marketingFee;

     

        player.dividends = 0;
        player.match_bonus = 0;
        player.total_withdrawn += withdrawAmount;
        withdrawn += withdrawAmount;
        marketing_fees += marketingFee;


        payable(MARKETING_WALLET).transfer(marketingFee);
        payable(msg.sender).transfer(withdrawAmount);
        
        emit Withdraw(msg.sender, withdrawAmount);
    }

    /**
     * @dev Calculates the current ROI earnings for a user
     * @param _addr User address to calculate earnings for
     * @return value Total pending ROI earnings in ETH
     *
     * Calculation Method:
     * - 1% daily ROI (0.01 = 10/1000)
     * - Prorated to the second (86400 seconds per day)
     * - Calculated separately for each active deposit
     * - Time period: from last payout (or deposit time) to current block time
     * - Automatically handles multiple deposits with different start times
     */
    function payoutOf(address _addr) view external returns(uint256 value) {
        Player storage player = players[_addr];

        for(uint256 i = 0; i < player.deposits.length; i++) {
            Deposit storage dep = player.deposits[i];

            uint256 from = player.last_payout > dep.time ? player.last_payout : dep.time;
            uint256 to = block.timestamp;

            if(from < to) {
                // Calculate 1% daily ROI
                // 1% = 0.01 = 10/1000
                // 86400 seconds in a day
                value += (dep.amount * (to - from) * 10) / (86400 * 1000);
            }
        }

        return value;
    }
    
    /**
     * @dev Retrieves comprehensive information about a user's account
     * @param _addr User address to get information for
     * @return for_withdraw Total available to withdraw (ROI + dividends + match bonus)
     * @return total_invested Total amount invested by user
     * @return total_withdrawn Total amount withdrawn by user
     * @return total_match_bonus Total referral bonuses earned
     * @return structure Array of referral counts at each level [L1, L2, L3, L4, L5]
     *
     * This function aggregates all user statistics including:
     * - Current ROI earnings
     * - Investment history
     * - Withdrawal history
     * - Referral earnings
     * - Referral structure
     */
    function userInfo(address _addr) view external returns(uint256 for_withdraw, uint256 total_invested, uint256 total_withdrawn, uint256 total_match_bonus, uint256[BONUS_LINES_COUNT] memory structure) {
        Player storage player = players[_addr];

        uint256 payout = this.payoutOf(_addr);

        for(uint8 i = 0; i < ref_bonuses.length; i++) {
            structure[i] = player.structure[i];
        }

        return (
            payout + player.dividends + player.match_bonus,
            player.total_invested,
            player.total_withdrawn,
            player.total_match_bonus,
            structure
        );
    }

    /**
     * @dev Retrieves global contract statistics
     * @return _invested Total ETH invested in the protocol
     * @return _withdrawn Total ETH withdrawn from the protocol
     * @return _match_bonus Total referral bonuses paid out
     *
     * This function provides transparency into the protocol's overall metrics
     * All values are cumulative since contract deployment
     */
    function contractInfo() view external returns(uint256 _invested, uint256 _withdrawn, uint256 _match_bonus) {
        return (invested, withdrawn, match_bonus);
    }

    /**
     * @dev Retrieves all stakers count
     * @return Total stakers count
     */
    function getTotalStakers() external view returns (uint256) {
        return stakers.length;
    }
}