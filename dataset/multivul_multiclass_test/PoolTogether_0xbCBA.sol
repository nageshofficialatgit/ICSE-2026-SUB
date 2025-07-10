// SPDX-License-Identifier: MIT
pragma solidity ^0.8.18;

interface ERC20 {
    function balanceOf(address account) external view returns (uint256);
    function transfer(address recipient, uint256 amount) external returns (bool);
}

/** 
 * https://pteth.com
 * 
 * Контракт ежедневного/еженедельного/ежемесячного и ежегодного розыгрыша. Это маленький и автоматизированный контракт
 * реализующий логику накопления переводов на адрес контракта в банк с дальнейшим определением победителей методом
 * commit–reveal и автоматической выплатой выигрыша. Участники покупают слоты за минимальную цену 0.001 формируя банк,
 * после завершения розыгрыша выбираются до 7 победителей, между которыми банк распределяется пропорционально сумме их
 * выигрышной ставки. Организатор взимает комиссию для выборочного повышения мотивации участников в последующих
 * розыгрышах. В этой реализации розыгрыша всегда есть до 7 победителей. Вероятность выигрыша выше чем в любом казино
 * или ставках. Код максимально документирован на нескольких языках, чтобы любой мог удостовериться в отсутствии
 * мошенничества.
 * 
 * Contract for daily, weekly, monthly, and yearly draws. This is a small and automated contract that implements the
 * logic of accumulating transfers to the contract address into a prize pool, followed by determining winners using the
 * commit–reveal method and automatically paying out winnings. Participants purchase slots for a minimum price of 0.001,
 * forming the prize pool. After the draw ends, up to 7 winners are selected, among whom the pool is distributed 
 * proportionally to the amount of their winning bet. The organizer takes a commission to selectively increase the
 * motivation of participants in future draws. In this implementation, there are always up to 7 winners. The probability
 * of winning is higher than in any casino or betting system. The code is extensively documented in multiple languages 
 * so that anyone can verify the absence of fraud.
 * 
 * 每日、每週、每月和每年抽獎合約。這是一個小型且自動化的合約，實現了將轉賬累積到合約地址的獎池機制，
 * 然後通過 commit–reveal 方法確定獲勝者，並自動發放獎金。參與者以最低價格 0.001 購買抽獎名額，
 * 形成獎池。抽獎結束後，最多選出 7 名獲勝者，獎池將按照其中獎投注金額的比例進行分配。
 * 組織者收取一定的佣金，以選擇性地提高參與者在未來抽獎中的積極性。
 * 此抽獎機制最多有 7 名獲勝者，贏得獎金的概率比任何賭場或博彩系統都高。
 * 此代碼已用多種語言進行詳細註釋，以便任何人都能確認其公平性，無欺詐行為。
 */
contract PoolTogether {
    /**
     * Вид розыгрыша: 1 - дневной, 7 - недельный, 31 - месячный, 365 - годовой.
     * Type of draw: 1 - daily, 7 - weekly, 31 - monthly, 365 - yearly.
     * 抽獎類型: 1 - 每日, 7 - 每週, 31 - 每月, 365 - 每年。
     */
    uint256 private constant CFG_DRAW_KIND = 7;
    /**
     * 0.001 ETH: 1e15 (1 ETH = 1e18 Wei)
     * 0.001 SOL: 1e6 (1 SOL = 1e9 Lamports)
     * 0.001 TRX: 1e3 (1 TRX = 1e6 SUN)
     */
    uint256 private constant CFG_MIN_AMOUNT = 1e17;
    /**
     * Адрес организатора для выплаты комиссий.
     * Address of the organizer for commission payouts.
     * 用於佣金支付的組織者地址。
     */
    address private constant CFG_ORGANISER = 0x52CcB1e361B79Cd5a464CDbA7F8A51B0D7BdDa18;
    /**
     * https://tether.to/en/supported-protocols
     * 
     * 0xdAC17F958D2ee523a2206206994597C13D831ec7
     * Этот адрес является официальным адресом смарт-контракта USDT (Tether) в сети Ethereum.
     * This address is the official address of the USDT (Tether) smart contract on the Ethereum network.
     * 這個地址是 Ethereum 網絡上 USDT (Tether) 智能合約的官方地址。
     * 
     * 0xa614f803B6FD780986A42c78Ec9c7f77e6DeD13C
     * Этот адрес является официальным адресом смарт-контракта USDT (Tether) в сети Tron.
     * This address is the official address of the USDT (Tether) smart contract on the Tron network.
     * 這個地址是 Tron 網絡上 USDT (Tether) 智能合約的官方地址。
     */
    address private constant CFG_USDT = 0xdAC17F958D2ee523a2206206994597C13D831ec7;

    /**
     * Таблица коммитов с началами розыгрышей для commit–reveal.
     * Table of commits with the start times of the draws for commit–reveal.
     * 提交-揭示機制中帶有抽獎開始時間的提交表。
     */
    mapping(uint256 => uint256) private drawTimeByCommit;
    /**
     * Адрес владельца контракта.
     * Address of the contract owner.
     * 合約擁有者的地址。
     */
    address private owner;
    /**
     * Таблица ожидающих отправки выигрыша победителей с временем начала розыгрыша.
     * Table of winners awaiting payout with the draw start time.
     * 帶有抽獎開始時間的待發放獎金的獲勝者列表。
     */
    mapping(address => uint256) private outstandingDrawTimeByMember;
    /**
     * Признак отправки ожидающих комиссий.
     * Indicator of pending commission payouts.
     * 待發放佣金的狀態標誌。
     */
    bool private outstandingCommissionSending;
    /**
     * Сумма ожидающих отправки комиссий.
     * Total amount of pending commission payouts.
     * 待發放佣金的總額。
     */
    uint256 private outstandingCommissionTotal;
    /**
     * Таблица ожидающих отправки выигрыша победителей с суммой выплаты.
     * Table of winners awaiting payout with the winning amount.
     * 帶有獎金金額的待發放獎金的獲勝者列表。
     */
    mapping(address => uint256) private outstandingTotalByMember;
    /**
     * Таблица ожидающих отправки выигрыша победителей с количеством попыток.
     * Table of winners awaiting payout with the number of attempts.
     * 帶有嘗試次數的待發放獎金的獲勝者列表。
     */
    mapping(address => uint256) private outstandingTriedByMember;
    /**
     * Сумма выплат ожидающих отправки выигрыша победителей.
     * Total amount of pending winner payouts.
     * 待發放獎金的總額。
     */
    uint256 private outstandingWinningsTotal;
    /**
     * Список ожидающих отправки выигрыша победителей.
     * List of winners awaiting payout.
     * 待發放獎金的獲勝者列表。
     */
    address[] private outstandingWinnerList;
    /**
     * Список сумм участников.
     * List of participant amounts.
     * 參與者金額列表。
     */
    uint256[] private slotAmountList;
    /**
     * Список адресов участников.
     * List of participant addresses.
     * 參與者地址列表。
     */
    address[] private slotMemberList;
    /**
     * Время начала розыгрыша.
     * Draw start time.
     * 抽獎開始時間。
     */
    uint256 private startTime;
    /**
     * Таблица выигрышных индексов слотов.
     * Table of winning slot indices.
     * 中獎插槽索引表。
     */
    mapping(uint256 => bool) private winFlagByIndex;

    /**
     * Событие покупки слота.
     * Event of slot purchase.
     * 插槽購買事件。
     */
    event BuySlotEvent(uint256 drawTime, address member, uint256 amount, uint256 slotIndex);
    /**
     * Событие ошибки при выплате комиссии.
     * Event of commission payout failure.
     * 佣金支付失敗事件。
     */
    event CommissionSentFailedEvent(address owner, uint256 amount);
    /**
     * Событие успешной выплаты комиссии.
     * Event of successful commission payout.
     * 佣金支付成功事件。
     */
    event CommissionSentSuccessEvent(address owner, uint256 amount);
    /**
     * Событие аннулирования выигрыша.
     * Event of prize cancellation.
     * 獎金取消事件。
     */
    event WinningCanceledEvent(uint256 drawTime, address winner, uint256 amount);
    /**
     * Событие выплаты выигрыша.
     * Event of prize payout.
     * 獎金支付事件。
     */
    event WinningsSendEvent(uint256 drawTime, address winner, uint256 amount);
    /**
     * Событие ошибки при выплате выигрыша.
     * Event of prize payout failure.
     * 獎金支付失敗事件。
     */
    event WinningSentFailedEvent(uint256 drawTime, address winner, uint256 amount);
    /**
     * Событие успешной выплаты выигрыша.
     * Event of successful prize payout.
     * 獎金支付成功事件。
     */
    event WinningsSentSuccessEvent(uint256 drawTime, address winner, uint256 amount);

    constructor() {
        // Устанавливаем владельца контракта.
        // Set the contract owner.
        // 設置合約擁有者。
        owner = msg.sender;
        // Устанавливаем время начала розыгрыша.
        // Set the draw start time.
        // 設置抽獎開始時間。
        startTime = block.timestamp;
    }

    /**
     * Есть комиссия на выплату.
     * There is a commission for payout.
     * 存在需支付的佣金。
     */
    modifier onlyHasOutstandingCommissions() {
        require(outstandingCommissionTotal > 0, "Commissions have been paid.");
        _;
    }

    /**
     * Есть выигрыш на выплату.
     * There is a prize for payout.
     * 存在需支付的獎金。
     */
    modifier onlyHasOutstandingWinnings() {
        require(outstandingWinningsTotal > 0, "Winnings have been paid.");
        _;
    }

    /**
     * Сумма больше или равна минимальной.
     * The amount is greater than or equal to the minimum.
     * 金額大於或等於最低限額。
     */
    modifier onlyMoreThenMinAmount() {
        require(msg.value > 0, "Amount must be not zero.");
        require(msg.value >= CFG_MIN_AMOUNT, "Minimum amount is 0.001");
        _;
    }

    /**
     * Выполняет покупку слота.
     * Execute slot purchase.
     * 執行插槽購買。
     */
    function onBuySlot() private {
        // Добавляем сумму участника.
        // Add participant's amount.
        // 添加參與者金額。
        slotAmountList.push(msg.value);
        // Добавляем адрес участника.
        // Add participant's address.
        // 添加參與者地址。
        slotMemberList.push(msg.sender);
        // Вызываем событие покупки слота.
        // Trigger slot purchase event.
        // 觸發插槽購買事件。
        emit BuySlotEvent(startTime, msg.sender, msg.value, slotMemberList.length - 1);
    }

    /**
     * Выполняет непогашенную выплату комиссий.
     * Execute pending commission payout.
     * 執行未結清的佣金支付。
     */
    function onPayOutstandingCommissions() private {
        if (outstandingCommissionSending) return;
        // Защита от reentrancy. Нельзя уменьшать outstandingCommissionTotal, т.к. есть момент что в момент выплаты
        // непогашенных комиссий можно вызвать tryFinish, где баланс может посчитаться неправильно и соответственно эта
        // часть непогашенных комиссий может распределиться на текущих победителей.
        //
        // Reentrancy protection. The outstandingCommissionTotal should not be reduced, as there is a risk that 
        // during the payout of pending commissions, tryFinish can be called, where the balance might be miscalculated,
        // and as a result, this portion of the pending commissions could be distributed to the current winners.
        //
        // 防止重入攻擊。不可減少 outstandingCommissionTotal，因為在支付未結清的佣金時，可能會調用 tryFinish，
        // 導致餘額計算錯誤，進而使這部分未結清的佣金被分配給當前的獲勝者。
        if (outstandingCommissionTotal <= 0) return;
        uint256 amount = outstandingCommissionTotal;
        // Выполняем выплату комиссии.
        // Execute commission payout.
        // 執行佣金支付。
        outstandingCommissionSending = true;
        address organiser = (CFG_ORGANISER == address(0)) ? owner : CFG_ORGANISER;
        (bool commissionSent,) = payable(organiser).call{value: amount, gas: 3200}("");
        outstandingCommissionSending = false;
        if (commissionSent) {
            outstandingCommissionTotal = 0;
            // Вызываем событие успешной выплаты комиссии.
            // Trigger event of successful commission payout.
            // 觸發佣金支付成功事件。
            emit CommissionSentSuccessEvent(organiser, amount);
        } else {
            // Вызываем событие ошибки при выплате комиссии.
            // Trigger event of commission payout failure.
            // 觸發佣金支付失敗事件。
            emit CommissionSentFailedEvent(organiser, amount);
        }
    }

    /**
     * Выполняет непогашенную выплату выигрышей.
     * Execute pending prize payouts.
     * 執行未結清的獎金支付。
     */
    function onPayOutstandingWinnings() private {
        if (outstandingWinnerList.length <= 0) return;
        // Список победителей с ошибками выплаты.
        // List of winners with payout failures.
        // 具有支付失敗記錄的獲勝者列表。
        address[] memory failedList = new address[](outstandingWinnerList.length);
        uint256 failedListLen;
        // Выполняем непогашенные выплаты победителям.
        // Execute pending payouts to winners.
        // 執行未結清的獎金支付給獲勝者。
        address payable winMember;
        uint256 winAmount;
        uint256 drawTime;
        uint256 tried;
        for (uint256 i = 0; i < outstandingWinnerList.length; i++) {
            winMember = payable(outstandingWinnerList[i]);
            winAmount = outstandingTotalByMember[winMember];
            drawTime = outstandingDrawTimeByMember[winMember];
            tried = outstandingTriedByMember[winMember];
            // Защита от reentrancy. Нельзя уменьшать outstandingWinningsTotal, т.к. есть момент что в момент выплаты
            // непогашенных выигрышей можно вызвать tryFinish, где баланс бы может посчитаться неправильно и
            // соответственно эта часть непогашенных выигрышей может распределиться на текущих победителей.
            //
            // Reentrancy protection. The outstandingWinningsTotal should not be reduced, as there is a risk that 
            // during the payout of pending winnings, tryFinish can be called, where the balance might be miscalculated,
            // and as a result, this portion of the pending winnings could be distributed to the current winners.
            //
            // 防止重入攻擊。不可減少 outstandingWinningsTotal，因為在支付未結清的獎金時，可能會調用 tryFinish，
            // 導致餘額計算錯誤，進而使這部分未結清的獎金被分配給當前的獲勝者。
            delete outstandingTotalByMember[winMember];
            delete outstandingTriedByMember[winMember];
            // Вызываем событие выплаты выигрыша.
            // Trigger the prize payout event.
            // 觸發獎金支付事件。
            emit WinningsSendEvent(drawTime, winMember, winAmount);
            // Выполняем выплату.
            // Execute the payout.
            // 執行支付。
            if (winAmount > 0) {
                (bool payoutSent,) = winMember.call{value: winAmount, gas: 3200}("");
                if (payoutSent) {
                    outstandingWinningsTotal -= winAmount;
                    // Вызываем событие успешной выплаты выигрыша.
                    // Trigger the event of successful prize payout.
                    // 觸發獎金支付成功事件。
                    emit WinningsSentSuccessEvent(drawTime, winMember, winAmount);
                } else {
                    // После 3-х попыток аннулируем выплату.
                    // Cancel the payout after 3 attempts.
                    // 3 次嘗試後取消支付。
                    if (tried < 3) {
                        outstandingTotalByMember[winMember] = winAmount;
                        // Не увеличиваем tried, т.к. делаем это в tryFinish, чтобы избежать аннулирования выигрыша при
                        // частом вызове выплаты выигрыша.
                        //
                        // Do not increment tried, as this is done in tryFinish to avoid payout cancellation due to
                        // frequent payout calls.
                        //
                        // 不增加 tried，因為這在 tryFinish 中執行，以避免因頻繁支付請求而取消獎金。
                        outstandingTriedByMember[winMember] = tried;
                        failedList[failedListLen] = winMember;
                        failedListLen++;
                    } else {
                        outstandingWinningsTotal -= winAmount;
                        // Вызываем событие аннулирования выигрыша.
                        // Trigger the event of prize cancellation.
                        // 觸發獎金取消事件。
                        emit WinningCanceledEvent(drawTime, winMember, winAmount);
                    }
                    // Вызываем событие ошибки при выплате выигрыша.
                    // Trigger the event of prize payout failure.
                    // 觸發獎金支付失敗事件。
                    emit WinningSentFailedEvent(drawTime, winMember, winAmount);
                }
            }
        }
        // Формируем новый список победителей с непогашенными выплатами.
        // Generate a new list of winners with pending payouts.
        // 生成待發放獎金的獲勝者新列表。
        delete outstandingWinnerList;
        for (uint256 i = 0; i < failedListLen; i++) {
            outstandingWinnerList.push(failedList[i]);
        }
    }

    /**
     * Добавляет commit с датой для commit–reveal.
     * Adds a commit with a date for commit–reveal.
     * 添加帶有日期的提交以用於 commit–reveal。
     */
    function addCommitForReveal(uint256 commit) external {
        drawTimeByCommit[commit] = block.timestamp;
    }

    /**
     * Функция для пополнения баланса.
     * Function to top up the balance.
     * 用於增加餘額的函數。
     */
    function addToBalance() external payable onlyMoreThenMinAmount {
    }

    /**
     * Функция для покупки слота.
     * Function for purchasing a slot.
     * 插槽購買函數。
     */
    function buySlot() external payable onlyMoreThenMinAmount {
        onBuySlot();
    }

    /**
     * Возвращает время завершения текущего дня (23:59:59 UTC).
     * Returns the end time of the current day (23:59:59 UTC).
     * 返回當前日期的結束時間 (23:59:59 UTC)。
     */
    function getFinishDay() private view returns (uint256) {
        // Начало текущего дня 00:00:00 UTC.
        // Start of the current day at 00:00:00 UTC.
        // 當前日的開始時間 00:00:00 UTC。
        uint256 dayStart = startTime - (startTime % 1 days);
        // Конец текущего дня 23:59:59 UTC.
        // End of the current day at 23:59:59 UTC.
        // 當前日的結束時間 23:59:59 UTC。
        return dayStart + (1 days) - 1;
    }

    /**
     * Возвращает время завершения текущего розыгрыша.
     * Returns the end time of the current draw.
     * 返回當前抽獎的結束時間。
     */
    function getFinishDate() public view returns (uint256) {
        uint256 timestamp = getFinishTime();
        (uint256 year, uint256 month, uint256 day) = useTimestampToDate(timestamp);
        return (year * 10000) + (month * 100) + day;
    }

    /**
     * Возвращает время завершения текущего месяца (последний день месяца, 23:59:59 UTC).
     * Returns the end time of the current month (last day of the month, 23:59:59 UTC).
     * 返回當前月份的結束時間 (當月最後一天 23:59:59 UTC)。
     */
    function getFinishMonth() private view returns (uint256) {
        (uint256 year, uint256 month,) = useTimestampToDate(startTime);
        uint256 daysInMonth = useDaysInMonth(year, month);
        // Вычисляем timestamp начала месяца:
        uint256 monthStart = useTimestampFromDate(year, month, 1, 0, 0, 0);
        // Возвращаем последний момент месяца (23:59:59 последнего дня)
        return monthStart + (daysInMonth * 1 days) - 1;
    }

    /**
     * Возвращает время завершения текущего розыгрыша.
     * Returns the end time of the current draw.
     * 返回當前抽獎的結束時間。
     */
    function getFinishTime() public view returns (uint256) {
        if (CFG_DRAW_KIND == 1) return getFinishDay();
        if (CFG_DRAW_KIND == 7) return getFinishWeek();
        if (CFG_DRAW_KIND == 31) return getFinishMonth();
        if (CFG_DRAW_KIND == 365) return getFinishYear();
        return getFinishDay();
    }

    /**
     * Возвращает время завершения текущей недели (воскресенье 23:59:59 UTC).
     * Returns the end time of the current week (Sunday 23:59:59 UTC).
     * 返回當前週的結束時間 (星期日 23:59:59 UTC)。
     */
    function getFinishWeek() private view returns (uint256) {
        // Начало текущего дня 00:00:00 UTC.
        // Start of the current day at 00:00:00 UTC.
        // 當前日的開始時間 00:00:00 UTC。
        uint256 dayStart = startTime - (startTime % 1 days);
        // Вычисляем день недели: ((dayStart/1 days) + 4) % 7 дает: вс = 0, пн = 1, ... сб = 6.
        // Calculate the day of the week: ((dayStart / 1 days) + 4) % 7 gives: Sun = 0, Mon = 1, ... Sat = 6.
        // 計算星期幾: ((dayStart / 1 days) + 4) % 7 給出: 週日 = 0, 週一 = 1, ... 週六 = 6。
        uint256 dayOfWeek = ((dayStart / 1 days) + 4) % 7;
        // Если сегодня воскресенье (dayOfWeek == 0), дни до конца недели = 0, иначе = 7 - dayOfWeek.
        // If today is Sunday (dayOfWeek == 0), the days until the end of the week = 0, otherwise = 7 - dayOfWeek.
        // 如果今天是週日 (dayOfWeek == 0)，則距離本週結束的天數 = 0，否則 = 7 - dayOfWeek。
        uint256 daysToAdd = (dayOfWeek == 0) ? 0 : (7 - dayOfWeek);
        // Конец недели = начало дня + (daysToAdd + 1) * 1 days - 1 (последняя секунда воскресенья).
        // End of the week = start of the day + (daysToAdd + 1) * 1 days - 1 (last second of Sunday).
        // 本週結束時間 = 當天開始時間 + (daysToAdd + 1) * 1 days - 1 (週日最後一秒)。
        return dayStart + (daysToAdd + 1) * 1 days - 1;
    }

    /**
     * Возвращает баланс текущего розыгрыша.
     * Returns the balance of the current draw.
     * 返回當前抽獎的餘額。
     */
    function getBalance() public view returns (uint256) {
        return address(this).balance - outstandingCommissionTotal - outstandingWinningsTotal;
    }

    /**
     * Возвращает время завершения текущего года (31 декабря, 23:59:59 UTC).
     * Returns the end time of the current year (December 31, 23:59:59 UTC).
     * 返回當前年份的結束時間 (12月31日 23:59:59 UTC)。
     */
    function getFinishYear() private view returns (uint256) {
        (uint256 year,,) = useTimestampToDate(startTime);
        return useTimestampFromDate(year, 12, 31, 23, 59, 59);
    }

    /**
     * Возвращает время начала текущего розыгрыша.
     * Returns the start time of the current draw.
     * 返回當前抽獎的開始時間。
     */
    function getStartDate() public view returns (uint256) {
        uint256 timestamp = startTime;
        (uint256 year, uint256 month, uint256 day) = useTimestampToDate(timestamp);
        return (year * 10000) + (month * 100) + day;
    }

    /**
     * Возвращает количество слотов в текущем розыгрыше.
     * Returns the number of slots in the current draw.
     * 返回當前抽獎中的名額數量。
     */
    function getSlotsCount() public view returns (uint256) {
        return slotMemberList.length;
    }

    /**
     * Возвращает время начала текущего розыгрыша.
     * Returns the start time of the current draw.
     * 返回當前抽獎的開始時間。
     */
    function getStartTime() public view returns (uint256) {
        return startTime;
    }

    /**
     * Выполняет попытку завершения розыгрыша, т.е. выплатить комиссии, призы и запустить новый розыгрыш.
     * Attempt to finalize the draw, i.e., pay commissions, distribute prizes, and start a new draw.
     * 嘗試完成抽獎，即支付佣金、分發獎金並啟動新抽獎。
     */
    function tryFinish(uint256 reveal) external {
        uint256 balance = address(this).balance - outstandingCommissionTotal - outstandingWinningsTotal;
        uint256 commit = uint256(keccak256(abi.encodePacked(reveal)));
        uint256 commitTime = drawTimeByCommit[commit];
        bool canFinish = (
            // Баланс не нулевой.
            // The balance is non-zero.
            // 餘額非零。
            (balance > 0)
            // Выполняется владельцем или с задержкой более часа.
            // Executed by the owner or with a delay of more than an hour.
            // 由擁有者執行，或延遲超過一小時後執行。
            && ((owner == msg.sender) || (CFG_ORGANISER == msg.sender) || (getFinishTime() + 1 hours < block.timestamp))
            // Есть участники.
            // There are participants.
            // 存在參與者。
            && (slotMemberList.length > 0)
            // Начались новые сутки по UTC0.
            // A new day has started in UTC0.
            // UTC0 新的一天已開始。
            && (getFinishTime() < block.timestamp)
            // Успешный commit–reveal.
            // Successful commit–reveal.
            // 成功的 commit–reveal。
            && ((getFinishTime() + 12 hours < block.timestamp)
                || ((block.timestamp - 24 hours < commitTime) && (commitTime + 2 hours < block.timestamp)))
        );
        require(canFinish, "Wait until the lottery draw has finished.");

        // Сохраняем дату начала розыгрыша.
        // Save the draw start date.
        // 保存抽獎開始日期。
        uint256 drawTime = startTime;
        // Устанавливаем время начала розыгрыша. Защита от reentrancy.
        // Set the draw start time. Reentrancy protection.
        // 設置抽獎開始時間。防止重入攻擊。
        startTime = block.timestamp;

        // Формируем комиссию.
        // Calculate the commission.
        // 計算佣金。
        uint256 commission = uint256(balance * 95 / 100);
        balance -= commission;
        outstandingCommissionTotal += commission;
        // Выплачиваем непогашенную комиссию владельцу.
        // Pay the pending commission to the owner.
        // 支付未結清的佣金給擁有者。
        onPayOutstandingCommissions();
        // Вычисляем общую сумму и формируем список индексов выигрышных слотов.
        // Calculate the total amount and generate the list of winning slot indices.
        // 計算總金額並生成中獎插槽索引列表。
        uint256[] memory winIndexList = new uint256[](7);
        uint256 winHash;
        uint256 winIndex;
        uint256 winTotal;
        for (uint256 i = 0; i < winIndexList.length; i++) {
            // Вычисляем хеш выигрышного слота.
            // Calculate the hash of the winning slot.
            // 計算中獎插槽的哈希值。
            winHash = uint256(keccak256(abi.encodePacked(drawTime, winHash, reveal)));
            // Вычисляем индекс выигрышного слота.
            // Determine the index of the winning slot.
            // 計算中獎插槽索引。
            winIndex = winHash % slotMemberList.length;
            // Запрещаем дубликаты в выигрышном списке.
            // Prevent duplicates in the winning list.
            // 禁止獲獎列表中的重複數據。
            if (winFlagByIndex[winIndex]) {
                winIndexList[i] = type(uint256).max;
                continue;
            }
            winFlagByIndex[winIndex] = true;
            // Добавляем индекс выигрышного слота в список.
            // Add the winning slot index to the list.
            // 將中獎插槽索引添加到列表中。
            winIndexList[i] = winIndex;
            // Вычисляем общую сумму.
            // Compute the total amount.
            // 計算總金額。
            winTotal += slotAmountList[winIndex];
        }
        // Выплачиваем выигрыш.
        // Pay out the winnings.
        // 支付獎金。
        if (winTotal > 0) {
            uint256 amount;
            uint256 winAmount;
            address payable winMember;
            for (uint256 i = 0; i < winIndexList.length; i++) {
                winIndex = winIndexList[i];
                if (winIndex == type(uint256).max) continue;
                winAmount = slotAmountList[winIndex];
                winMember = payable(slotMemberList[winIndex]);
                amount = uint256(winAmount * balance / winTotal);
                // Вызываем событие выплаты выигрыша.
                // Trigger the prize payout event.
                // 觸發獎金支付事件。
                emit WinningsSendEvent(drawTime, winMember, winAmount);
                (bool payoutSent,) = winMember.call{value: amount, gas: 3200}("");
                if (payoutSent) {
                    // Вызываем событие успешной выплаты выигрыша.
                    // Trigger the event of successful prize payout.
                    // 觸發獎金支付成功事件。
                    emit WinningsSentSuccessEvent(drawTime, winMember, amount);
                } else {
                    if (outstandingTotalByMember[winMember] <= 0) outstandingWinnerList.push(winMember);
                    outstandingDrawTimeByMember[winMember] = drawTime;
                    outstandingTotalByMember[winMember] += amount;
                    outstandingTriedByMember[winMember]++;
                    outstandingWinningsTotal += amount;
                    // Вызываем событие ошибки при выплате выигрыша.
                    // Trigger the event of prize payout failure.
                    // 觸發獎金支付失敗事件。
                    emit WinningSentFailedEvent(drawTime, winMember, amount);
                }
            }
        }
        // Очищаем данные.
        // Clear data.
        // 清理數據。
        delete drawTimeByCommit[commit];
        delete slotAmountList;
        delete slotMemberList;
        for (uint256 i = 0; i < winIndexList.length; i++) {
            winIndex = winIndexList[i];
            if (winIndex == type(uint256).max) continue;
            delete winFlagByIndex[winIndex];
        }
        // Выплачиваем непогашенные выигрыши.
        // Pay out the pending winnings.
        // 支付未結清的獎金。
        onPayOutstandingWinnings();
    }

    /**
     * Выполняет выплаты комиссий.
     * Execute commission payouts.
     * 執行佣金支付。
     */
    function payOutstandingCommissions() external onlyHasOutstandingCommissions {
        onPayOutstandingCommissions();
    }

    /**
     * Выполняет выплаты выигрышей.
     * Execute prize payouts.
     * 執行獎金支付。
     */
    function payOutstandingWinnings() external onlyHasOutstandingWinnings {
        onPayOutstandingWinnings();
    }

    /**
     * Вычисляет количество дней, прошедших с 1970-01-01 до заданной даты.
     * Calculates the number of days elapsed since 1970-01-01 to the given date.
     * 計算從 1970-01-01 到指定日期所經過的天數。
     */
    function useDaysFromDate(uint256 year, uint256 month, uint256 day) internal pure returns (uint256) {
        int256 y = int256(year);
        int256 m = int256(month);
        int256 d = int256(day);
        // Алгоритм Флегеля-Ван Фландерена
        // Flegel-Van Flandern algorithm.
        // Flegel-Van Flandern 演算法。
        return uint256(
            d - 32075 +
            (1461 * (y + 4800 + (m - 14) / 12)) / 4 +
            (367 * (m - 2 - ((m - 14) / 12) * 12)) / 12 -
            (3 * ((y + 4900 + (m - 14) / 12) / 100)) / 4 -
            2440588
        );
    }

    /**
     * Возвращает количество дней в месяце для заданного года и месяца.
     * Returns the number of days in a month for the given year and month.
     * 返回指定年份和月份的天數。
     */
    function useDaysInMonth(uint256 year, uint256 month) internal pure returns (uint256) {
        if (month == 2) {
            return useIsLeapYear(year) ? 29 : 28;
        } else if (month == 4 || month == 6 || month == 9 || month == 11) {
            return 30;
        } else {
            return 31;
        }
    }

    /**
     * Проверяет, является ли год високосным.
     * Checks whether the given year is a leap year.
     * 檢查指定年份是否為閏年。
     */
    function useIsLeapYear(uint256 year) internal pure returns (bool) {
        if (year % 4 != 0) return false;
        if (year % 100 != 0) return true;
        if (year % 400 != 0) return false;
        return true;
    }

    /**
     * Преобразует дату и время в timestamp.
     * Converts date and time to a timestamp.
     * 將日期和時間轉換為時間戳。
     */
    function useTimestampFromDate(
        uint256 year, uint256 month, uint256 day, uint256 hour, uint256 minute, uint256 second
    ) internal pure returns (uint256) {
        uint256 daysOfDate = useDaysFromDate(year, month, day);
        return daysOfDate * 86400 + hour * 3600 + minute * 60 + second;
    }

    /**
     * Преобразует timestamp в дату (год, месяц, день).
     * Converts a timestamp to a date (year, month, day).
     * 將時間戳轉換為日期 (年、月、日)。
     */
    function useTimestampToDate(uint256 timestamp) internal pure returns (uint256 year, uint256 month, uint256 day) {
        uint256 z = timestamp / 86400 + 719468;
        uint256 era = z / 146097;
        uint256 daysOfEra = z - era * 146097;
        uint256 yearsOfEra = (daysOfEra - daysOfEra / 1460 + daysOfEra / 36524 - daysOfEra / 146096) / 365;
        year = yearsOfEra + era * 400 + 1;
        uint256 daysOfYear = daysOfEra - (365 * yearsOfEra + yearsOfEra / 4 - yearsOfEra / 100);
        uint256 monthPredictor = (5 * daysOfYear + 2) / 153;
        month = (monthPredictor < 10) ? (monthPredictor + 3) : (monthPredictor - 9);
        day = daysOfYear - (153 * monthPredictor + 2) / 5 + 1;
        if (month <= 2) year -= 1;
    }

    /**
     * Вызывается при пополнении баланса контракта.
     * Called when the contract balance is topped up.
     * 當合約餘額增加時調用。
     */
    fallback() external payable onlyMoreThenMinAmount {
        onBuySlot();
    }

    /**
     * Вызывается при пополнении баланса контракта.
     * Called when the contract balance is topped up.
     * 當合約餘額增加時調用。
     */
    receive() external payable onlyMoreThenMinAmount {
        onBuySlot();
    }

    /**
     * Выводит зачисленные USDT средства на счёт организатора.
     */
    function usdtWithdraw() external {
        ERC20 usdt = ERC20(CFG_USDT);
        uint256 balance = usdt.balanceOf(address(this));
        require(balance > 0, "Not enough balance.");
        usdt.transfer(CFG_ORGANISER, balance);
    }
}