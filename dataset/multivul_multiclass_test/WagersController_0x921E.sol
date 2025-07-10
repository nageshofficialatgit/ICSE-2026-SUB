/**
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
pragma solidity ^0.8.24;

interface IERC20 {
    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
    function totalSupply() external view returns (uint256);
    function balanceOf(address account) external view returns (uint256);
    function transfer(address to, uint256 value) external returns (bool);
    function allowance(address owner, address spender) external view returns (uint256);
    function approve(address spender, uint256 value) external returns (bool);
    function transferFrom(address from, address to, uint256 value) external returns (bool);
}

abstract contract Context {
    function _msgSender() internal view virtual returns (address) {
        return msg.sender;
    }

    function _msgData() internal view virtual returns (bytes calldata) {
        return msg.data;
    }

    function _contextSuffixLength() internal view virtual returns (uint256) {
        return 0;
    }
}

abstract contract Ownable is Context {
    address private _owner;

    error OwnableUnauthorizedAccount(address account);
    error OwnableInvalidOwner(address owner);
    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);

    constructor(address initialOwner) {
        if (initialOwner == address(0)) {
            revert OwnableInvalidOwner(address(0));
        }
        _transferOwnership(initialOwner);
    }

    modifier onlyOwner() {
        _checkOwner();
        _;
    }

    function owner() public view virtual returns (address) {
        return _owner;
    }

    function _checkOwner() internal view virtual {
        if (owner() != _msgSender()) {
            revert OwnableUnauthorizedAccount(_msgSender());
        }
    }

    function renounceOwnership() public virtual onlyOwner {
        _transferOwnership(address(0));
    }

    function transferOwnership(address newOwner) public virtual onlyOwner {
        if (newOwner == address(0)) {
            revert OwnableInvalidOwner(address(0));
        }
        _transferOwnership(newOwner);
    }

    function _transferOwnership(address newOwner) internal virtual {
        address oldOwner = _owner;
        _owner = newOwner;
        emit OwnershipTransferred(oldOwner, newOwner);
    }
}

contract WagersController is Ownable {
    enum Outcome { NotSet, Cancelled, Draw, Win }

    struct Better {
        mapping(Outcome => mapping(string => uint256)) bet;
        uint256 totalAmount;
        uint256 payouts;
    }

    struct Payout {
        uint256 amount;
        uint256 fee;
    }

    struct Wager {
        string eventName;
        uint256 eventStartTime;
        string[] participants;
        Outcome outcome;
        string winner;
        mapping(IERC20 => mapping(address => Better)) betters;
        mapping(IERC20 => mapping(Outcome => mapping(string => uint256))) totalOutcomeAmount;
    }

    event WagerCreated(uint256 eventId, string eventName);
    event BetPlaced(uint256 eventId, address indexed better, IERC20 token, Outcome outcome, uint256 amount);
    event BetRemoved(uint256 eventId, address indexed better, uint256 amount);
    event OutcomeAnnounced(Outcome outcome, string winner);
    event PayoutClaimed(address indexed better, uint256 eventId, IERC20 token, uint256 amount);

    uint256 private constant FEE_PARTS = 10_000;

    mapping(uint256 => Wager) private _wagers;

    IERC20 public nativeToken;
    uint256 public nonNativeTokenFeePercent;
    address public feeWallet;

    constructor() Ownable(_msgSender()) {
        feeWallet = _msgSender();
    }

    function setNativeToken(IERC20 nativeToken_) external onlyOwner {
        require(address(nativeToken_) != address(0), "Wrong token address");
        nativeToken = nativeToken_;
    }

    function setNonNativeTokenFeePercent(uint256 feePercent) external onlyOwner {
        require(feePercent <= FEE_PARTS, "Fee cannot be bigger than 100%!");
        nonNativeTokenFeePercent = feePercent;
    }

    function setFeeWallet(address feeWallet_) external onlyOwner {
        require(address(feeWallet_) != address(0), "Wrong wallet address");
        feeWallet = feeWallet_;
    }

    function createWager(
        uint256 eventId,
        string calldata eventName,
        uint256 eventStartTime,
        string[] calldata participants
    ) public onlyOwner {
        Wager storage _wager = _wagers[eventId];
        require(_wager.eventStartTime == 0, "Wager has already created!");
        require(eventStartTime > block.timestamp, "The event should start in the future!");
        require(participants.length >= 2, "There must be at least 2 participants!");
        _wager.eventName = eventName;
        _wager.eventStartTime = eventStartTime;
        for (uint256 i = 0; i < participants.length; i++) {
            _wager.participants.push(participants[i]);
        }
        emit WagerCreated(eventId, eventName);
    }

    function wager(uint256 wagerId) external view returns(
        string memory eventName, uint256 eventStartTime,
        string[] memory participants, Outcome outcome, string memory winner)
    {
        Wager storage _wager = _wagers[wagerId];
        return (_wager.eventName, _wager.eventStartTime, _wager.participants, _wager.outcome, _wager.winner);
    }

    function announceOutcome(uint256 eventId, Outcome outcome, string calldata winner) external onlyOwner {
        require(outcome != Outcome.NotSet, "Unacceptable state!");
        Wager storage _wager = _wagers[eventId];
        require(_wager.outcome == Outcome.NotSet, "Already completed!");
        _wager.outcome = outcome;
        if (outcome == Outcome.Win) {
            bool found = false;
            for (uint256 i = 0; i < _wager.participants.length; i++) {
                if (keccak256(abi.encodePacked(winner)) == keccak256(abi.encodePacked(_wager.participants[i]))) {
                    found = true;
                    break;
                }
            }
            require(found, "Participant not found");
            _wager.winner = winner;
        }
        emit OutcomeAnnounced(_wager.outcome, _wager.winner);
    }

    function placeBet(uint256 eventId, IERC20 token, uint256 amount, Outcome outcome, string memory winner) external {
        Wager storage _wager = _wagers[eventId];
        require(_wager.outcome == Outcome.NotSet, "Already completed!");
        require(_wager.eventStartTime > block.timestamp, "The event has already started!");
        require(outcome == Outcome.Draw || outcome == Outcome.Win, "Wrong outcome provided!");
        require(token.balanceOf(_msgSender()) >= amount, "Insufficient funds!");
        require(token.allowance(_msgSender(), address(this)) >= amount, "Insufficient allowance!");

        if (outcome == Outcome.Draw) {
            winner = "";
        } else {
            bool found = false;
            for (uint256 i = 0; i < _wager.participants.length; i++) {
                if (keccak256(abi.encodePacked(winner)) == keccak256(abi.encodePacked(_wager.participants[i]))) {
                    found = true;
                    break;
                }
            }
            require(found, "Participant not found");
        }

        _wager.betters[token][_msgSender()].bet[outcome][winner] += amount;
        _wager.betters[token][_msgSender()].totalAmount += amount;
        _wager.totalOutcomeAmount[token][outcome][winner] += amount;
        token.transferFrom(_msgSender(), address(this), amount);
    }

    function betAmount(uint256 eventId, IERC20 token, address better, Outcome outcome, string memory winner) public view returns(uint256) {
        Wager storage _wager = _wagers[eventId];
        return _wager.betters[token][better].bet[outcome][winner];
    }

    function betsTotalOutcomeAmount(uint256 eventId, IERC20 token, Outcome outcome, string memory winner) public view returns(uint256) {
        Wager storage _wager = _wagers[eventId];
        return _wager.totalOutcomeAmount[token][outcome][winner];
    }

    function betterTotalBetsAmount(uint256 eventId, IERC20 token, address better) public view returns(uint256) {
        Wager storage _wager = _wagers[eventId];
        return _wager.betters[token][better].totalAmount;
    }

    function betterPayouts(uint256 eventId, IERC20 token, address better) public view returns(uint256) {
        Wager storage _wager = _wagers[eventId];
        return _wager.betters[token][better].payouts;
    }

    function calcPayout(uint256 eventId, address better, IERC20 token) public view returns(Payout memory) {
        Payout memory payout;
        Wager storage _wager = _wagers[eventId];
        if (_wager.outcome == Outcome.Cancelled) {
            payout.amount = _wager.betters[token][better].totalAmount;
        } else {
            uint256 totalWinAmount;
            uint256 totalLossAmount;

            if (_wager.outcome == Outcome.Draw && _wager.betters[token][better].bet[Outcome.Draw][""] > 0) {
                totalWinAmount = _wager.totalOutcomeAmount[token][Outcome.Draw][""];
                for (uint8 i = 0; i < _wager.participants.length; i++) {
                    totalLossAmount += _wager.totalOutcomeAmount[token][Outcome.Win][_wager.participants[i]];
                }
                payout.amount = totalLossAmount * _wager.betters[token][better].bet[Outcome.Draw][""] / totalWinAmount;
                payout.amount += _wager.betters[token][better].bet[Outcome.Draw][""];
                if (payout.amount == 0) {
                    payout.amount = _wager.betters[token][better].bet[Outcome.Draw][""];
                } else if (token != nativeToken) {
                    payout.fee = payout.amount * nonNativeTokenFeePercent / FEE_PARTS;
                    payout.amount -= payout.fee;
                }
            } else if (_wager.outcome == Outcome.Win && _wager.betters[token][better].bet[Outcome.Win][_wager.winner] > 0) {
                totalWinAmount = _wager.totalOutcomeAmount[token][Outcome.Win][_wager.winner];
                for (uint8 i = 0; i < _wager.participants.length; i++) {
                    if (keccak256(abi.encodePacked(_wager.participants[i])) != keccak256(abi.encodePacked(_wager.winner))) {
                        totalLossAmount += _wager.totalOutcomeAmount[token][Outcome.Win][_wager.participants[i]];
                    }
                }
                totalLossAmount += _wager.totalOutcomeAmount[token][Outcome.Draw][""];
                payout.amount = totalLossAmount * _wager.betters[token][better].bet[Outcome.Win][_wager.winner] / totalWinAmount;
                payout.amount += _wager.betters[token][better].bet[Outcome.Win][_wager.winner];
                if (payout.amount == 0) {
                    payout.amount = _wager.betters[token][better].bet[Outcome.Win][_wager.winner];
                } else if (token != nativeToken) {
                    payout.fee = payout.amount * nonNativeTokenFeePercent / FEE_PARTS;
                    payout.amount -= payout.fee;
                }
            }
        }
        payout.amount -= _wager.betters[token][better].payouts;
        return payout;
    }

    function claimPayout(uint256 eventId, IERC20 token) public {
        Wager storage _wager = _wagers[eventId];
        require(_wager.outcome != Outcome.NotSet, "Not completed yet!");
        Payout memory payout = calcPayout(eventId, _msgSender(), token);
        require(payout.amount > 0, "Nothing to pay!");
        if (payout.fee > 0) {
            token.transfer(feeWallet, payout.fee);
        }
        _wager.betters[token][_msgSender()].payouts += payout.amount;
        token.transfer(_msgSender(), payout.amount);
        emit PayoutClaimed(_msgSender(), eventId, token, payout.amount);
    }
}